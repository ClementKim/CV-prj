import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size, patch_size, is_cls_token = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.is_cls_token = is_cls_token

        self.patchify = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size)
        self.linear = nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        if is_cls_token:
            self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))

        else:
            self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.patchify(x)
        x = torch.flatten(x, 2)
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)

        if self.is_cls_token:
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            x = torch.cat((cls, x), dim = 1)

        x = x + self.positional_embedding

        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        A = F.softmax(q @ k.transpose(2, 3) * self.scale, dim = -1)
        
        SA = (A @ v).transpose(1, 2).reshape(B, N, D)

        out = self.dropout(self.o(SA))

        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop):
        super(MultiHeadCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, auxiliary_tokens):
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(auxiliary_tokens).reshape(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(auxiliary_tokens).reshape(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

        A = F.softmax(q @ k.transpose(2, 3) * self.scale, dim = -1)

        CA = (A @ v).transpose(1, 2).reshape(B, -1, D)

        out = self.dropout(self.o(CA))

        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop, out_features = None):
        super().__init__()

        out_features = out_features if out_features is not None else in_features

        self.linear1 = nn.Linear(in_features = in_features, out_features = hidden_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.linear2 = nn.Linear(in_features = hidden_features, out_features = out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return self.dropout(x)
    
class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop, attn_drop):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(embed_dim = dim, num_heads = num_heads, drop = attn_drop)
        self.mlp = MLP(in_features = dim, hidden_features = int(dim * mlp_ratio), drop = drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop, attn_drop):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(embed_dim = dim, num_heads = num_heads, drop = attn_drop)
        self.mlp = MLP(in_features = dim, hidden_features = int(dim * mlp_ratio), drop = drop)

    def forward(self, x, auxiliary_tokens):
        x = x + self.attn(self.norm1(x), auxiliary_tokens)
        x = x + self.mlp(self.norm2(x))

        return x
    
class SelfExpert(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, depth = 2, is_cls_token = True):
        super(SelfExpert, self).__init__()

        self.features = self.embed_dim = embed_dim
        self.depth = depth

        # Replace PatchEmbedding with a linear projection layer
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.Sequential(*[
            SelfAttnBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)
            for _ in range(self.depth)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x input shape: [B, 1, embed_dim, embed_dim]
        x = x.squeeze(1)            # Transform to [B, embed_dim, embed_dim]
        x = self.input_proj(x)

        x = self.blocks(x)
        x = self.norm(x)

        # Replace lossy x[:, 0] with global mean pooling
        return x.mean(dim = 1)

class CrossExpert(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, depth = 2, is_cls_token = True):
        super(CrossExpert, self).__init__()

        self.features = self.embed_dim = embed_dim
        self.depth = depth

        # Replace PatchEmbedding with a linear projection layer
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)
            for _ in range(self.depth)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, auxiliary_tokens):
        # x input shape: [B, 1, embed_dim, embed_dim]
        x = x.squeeze(1)            # Transform to [B, embed_dim, embed_dim]
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, auxiliary_tokens)
        x = self.norm(x)

        # Replace lossy x[:, 0] with global mean pooling
        return x.mean(dim = 1)

class DecoderCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop):
        super(DecoderCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, context_tokens):               # x: [B, N, D] queries, context_tokens: [B, M, D] (per-sample, unlike the shared aux above)
        B, N, D = x.shape
        M = context_tokens.shape[1]

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context_tokens).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context_tokens).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        A = F.softmax(q @ k.transpose(2, 3) * self.scale, dim = -1)

        out = (A @ v).transpose(1, 2).reshape(B, N, D)

        return self.dropout(self.o(out))

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop, attn_drop):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim = dim, num_heads = num_heads, drop = attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = DecoderCrossAttention(embed_dim = dim, num_heads = num_heads, drop = attn_drop)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features = dim, hidden_features = int(dim * mlp_ratio), drop = drop)

    def forward(self, x, context_tokens):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context_tokens)
        x = x + self.mlp(self.norm3(x))

        return x

class TransformerDecoder(nn.Module):
    """Per-pixel segmentation head: ViT spatial tokens (queries) cross-attend to the global
    material descriptor (context tokens), then a per-token linear predicts the class; upsample to input size."""
    def __init__(self, embed_dim, num_classes, num_heads, mlp_ratio, drop, attn_drop, depth = 2, num_context = 16):
        super(TransformerDecoder, self).__init__()

        self.num_context = num_context
        self.to_context = nn.Linear(embed_dim, num_context * embed_dim)   # global descriptor -> context tokens
        self.blocks = nn.ModuleList([
            DecoderBlock(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, descriptor, spatial, out_size):
        B, D, h, w = spatial.shape

        tokens = spatial.flatten(2).transpose(1, 2)                          # [B, N, D] queries
        context_tokens = self.to_context(descriptor).reshape(B, self.num_context, D)   # [B, M, D] context tokens

        for block in self.blocks:
            tokens = block(tokens, context_tokens)

        logits = self.classifier(self.norm(tokens))                          # [B, N, num_classes]
        logits = logits.transpose(1, 2).reshape(B, -1, h, w)                 # [B, num_classes, h, w]

        return F.interpolate(logits, size = out_size, mode = 'bilinear', align_corners = False)