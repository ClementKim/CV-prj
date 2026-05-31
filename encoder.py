import torch
import torch.nn as nn

from transformer import PatchEmbedding, MLP, SelfAttnBlock

class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, depth = 12, is_cls_token = True):
        super(TransformerEncoder, self).__init__()

        self.features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.is_cls_token = is_cls_token
        self.depth = depth
        self.patch_embed = PatchEmbedding(in_channels = in_channels, embed_dim = embed_dim, img_size = img_size, patch_size = patch_size, is_cls_token = is_cls_token)
        self.blocks = nn.Sequential(*[
            SelfAttnBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)

        h, w = H // self.patch_size, W // self.patch_size

        if self.is_cls_token:
            cls_vector = x[:, 0]        # global summary token -> feeds the bilinear outer product
            patches = x[:, 1:]
        else:
            cls_vector = x.mean(dim = 1)
            patches = x

        spatial = patches.transpose(1, 2).reshape(B, self.embed_dim, h, w)   # spatial token grid [B, D, h, w] -> decoder
        return cls_vector, spatial
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:      # project identity when shape changes
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
            
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layers=(3, 4, 6, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        
        self.layer1 = self._make_layer(64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(256, 512, layers[3], stride=2)

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP(in_features=512, hidden_features=512 * 4, drop=0.1, out_features=out_channels)

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_c, out_c, stride=1))   # each block is a distinct module
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pooling(x).flatten(1)                  # global vector -> feeds the bilinear outer product
        return self.mlp(x)