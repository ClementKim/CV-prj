import torch
import torch.nn as nn

from transformer import PatchEmbedding, MLP, SelfAttnBlock

class TransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token = True):
        super(TransformerEncoder, self).__init__()

        self.features = self.embed_dim = embed_dim
        self.is_cls_token = is_cls_token
        self.depth = 12
        self.patch_embed = PatchEmbedding(in_channels = in_channels, embed_dim = embed_dim, img_size = img_size, patch_size = patch_size, is_cls_token = is_cls_token)
        self.blocks = nn.ModuleList(*[
            SelfAttnBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)

        if self.is_cls_token:
            return x[:, 0]
        
        return x.mean(dim = 1)
    
class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetEncoder, self).__init__()

        self.pooling = nn.MaxPool2d(stride = 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.residual_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.residual_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.residual_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.residual_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.residual_block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.residual_block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.residual_block7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.residual_block8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.mlp = MLP(in_features = 512, hidden_features = 512 * 4, out_features = out_channels)

    def forward(self, x):
        x = self.residual_block1(x) # 7x7 conv, 64, /2
        x = self.pooling(x) # pool, /2

        for _ in range(6):
            x = self.residual_block2(x) # 3x3 conv, 64

        x = self.residual_block3(x) # 3x3 conv, 128, /2

        for _ in range(7):
            x = self.residual_block4(x) # 3x3 conv, 128

        x = self.residual_block5(x) # 3x3 conv, 256, /2

        for _ in range(11):
            x = self.residual_block6(x) # 3x3 conv, 256

        x = self.residual_block7(x) # 3x3 conv, 512, /2

        for _ in range(5):
            x = self.residual_block8(x) # 3x3 conv, 512

        x = self.avg_pooling(x).squeeze()

        return self.mlp(x)
