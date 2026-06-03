"""Pretrained dual encoder for the proposed material-segmentation model.

The original encoders were trained from scratch, which on ~1.4k MINC images simply
memorized the train set (train loss collapsed while val mIoU stayed flat). Here both
branches are ImageNet-pretrained so the model starts from real visual features:

    ViT branch    : timm ViT-S/16, gives a global CLS summary + a 32x32 patch grid.
    ResNet branch : torchvision ResNet-34, gives a global vector AND dense multi-scale
                    feature maps (stride 4/8/16) that the decoder upsamples through.

``DualEncoder`` projects every backbone output to a single working ``embed_dim`` so the
bilinear / MoE / decoder dims are decoupled from the (fixed) backbone widths.
"""

import timm
import torch
import torch.nn as nn

from torchvision.models import resnet34, ResNet34_Weights


class ViTBackbone(nn.Module):
    """Pretrained ViT-S/16. Returns (cls [B, Dv], grid [B, Dv, gh, gw])."""

    def __init__(self, img_size, pretrained = True):
        super().__init__()
        # img_size != 384 is fine: timm interpolates the pretrained positional embedding.
        self.vit = timm.create_model(
            'vit_small_patch16_384', pretrained = pretrained,
            img_size = img_size, num_classes = 0)
        self.embed_dim = self.vit.embed_dim                 # 384
        self.grid = self.vit.patch_embed.grid_size          # (32, 32) at img_size 512

    def forward(self, x):
        feats = self.vit.forward_features(x)                # [B, 1 + N, Dv] (cls at 0)
        cls = feats[:, 0]                                   # [B, Dv]
        patches = feats[:, 1:]                              # [B, N, Dv]
        B, N, Dv = patches.shape
        gh, gw = self.grid
        grid = patches.transpose(1, 2).reshape(B, Dv, gh, gw)
        return cls, grid


class ResNetBackbone(nn.Module):
    """Pretrained ResNet-34. Returns the multi-scale maps + a global vector.

    Keeping the dense maps (instead of only the global avg-pool the old encoder used)
    is what lets the decoder recover spatial resolution for segmentation.
    """

    channels = {'f1': 64, 'f2': 128, 'f3': 256, 'f4': 512}

    def __init__(self, pretrained = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        rn = resnet34(weights = weights)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)                                 # [B, 64, 128, 128]  stride 4
        f2 = self.layer2(f1)                                # [B, 128, 64, 64]   stride 8
        f3 = self.layer3(f2)                                # [B, 256, 32, 32]   stride 16
        f4 = self.layer4(f3)                                # [B, 512, 16, 16]   stride 32
        g = f4.mean(dim = (2, 3))                           # [B, 512] global vector
        return {'f1': f1, 'f2': f2, 'f3': f3, 'g': g}


class DualEncoder(nn.Module):
    """ViT + ResNet, every output projected to the shared working ``embed_dim``.

    Drop-in replacement for the old from-scratch encoder: returns exactly the three
    tensors the rest of the (unchanged) pipeline consumes.

    forward(x) -> (cls, res_global, spatial)
        cls         : [B, D]            ViT global summary   -> bilinear branch
        res_global  : [B, D]            ResNet global vector -> bilinear branch
        spatial     : [B, D, 32, 32]    fused ViT grid + ResNet stride-16 map -> decoder
    """

    def __init__(self, img_size, embed_dim, pretrained = True):
        super().__init__()
        self.vit = ViTBackbone(img_size, pretrained = pretrained)
        self.res = ResNetBackbone(pretrained = pretrained)

        Dv = self.vit.embed_dim
        ch = self.res.channels

        self.cls_proj = nn.Linear(Dv, embed_dim)
        self.res_global_proj = nn.Linear(ch['f4'], embed_dim)

        self.vit_grid_proj = nn.Conv2d(Dv, embed_dim, kernel_size = 1)
        self.f3_proj = nn.Conv2d(ch['f3'], embed_dim, kernel_size = 1)

    def forward(self, x):
        cls, grid = self.vit(x)
        res = self.res(x)

        cls = self.cls_proj(cls)                            # [B, D]
        res_global = self.res_global_proj(res['g'])         # [B, D]

        # Fuse the two stride-16 spatial views (both 32x32) into the decoder input.
        spatial = self.vit_grid_proj(grid) + self.f3_proj(res['f3'])     # [B, D, 32, 32]
        return cls, res_global, spatial
