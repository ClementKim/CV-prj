import os
import torch
import random
import argparse

import numpy as np
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader, random_split

from encoder import TransformerEncoder, ResNetEncoder
from transformer import SelfExpert, CrossExpert, TransformerDecoder
from preprocessing import MINCSegmentationDataset, seg_collate_fn, NUM_CLASSES, IGNORE_INDEX, IMG_SIZE

class Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token = True):
        super(Encoder, self).__init__()

        self.transformer_encoder = TransformerEncoder(img_size = img_size, patch_size = patch_size, in_channels = in_channels, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop, is_cls_token = is_cls_token)
        self.resnet_encoder = ResNetEncoder(in_channels = in_channels, out_channels = embed_dim)

    def forward(self, x):
        return self.transformer_encoder(x), self.resnet_encoder(x)

class DenseMoE(nn.Module):
    def __init__(self, self_expert_pool, cross_expert_pool, aux):
        super(DenseMoE, self).__init__()

        self.self_expert_pool = self_expert_pool
        self.cross_expert_pool = cross_expert_pool
        self.aux = nn.Parameter(aux)

    def forward(self, bilinear_feature):
        self_expert_output = [expert(bilinear_feature) for expert in self.self_expert_pool]
        cross_expert_output = [expert(bilinear_feature, self.aux) for expert in self.cross_expert_pool]

        self_agg = sum(self_expert_output)
        cross_agg = sum(cross_expert_output)

        return self_agg + cross_agg

class ProposedMethod(nn.Module):
    def __init__(self, encoder, self_expert_pool, cross_expert_pool, aux, decoder):
        super(ProposedMethod, self).__init__()

        self.encoder = encoder
        self.moe = DenseMoE(self_expert_pool, cross_expert_pool, aux)
        self.decoder = decoder

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        (cls_vector, spatial), resnet_feature = self.encoder(x)

        bilinear_feature = torch.bmm(cls_vector.unsqueeze(2), resnet_feature.unsqueeze(1)).unsqueeze(1) # (B, 1, embed_dim, embed_dim)

        descriptor = self.moe(bilinear_feature)             # global material descriptor [B, embed_dim]

        return self.decoder(descriptor, spatial, (H, W))    # [B, num_classes, H, W]

def train_step(model, batch, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()

    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    loss = loss_fn(model(images), labels)

    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate(model, loader, num_classes, device):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype = torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim = 1)

            valid = labels != IGNORE_INDEX                  # exclude void pixels
            target = labels[valid].view(-1)
            pred = preds[valid].view(-1)

            idx = target * num_classes + pred
            confusion += torch.bincount(idx.cpu(), minlength = num_classes ** 2).reshape(num_classes, num_classes)

    total = confusion.sum().item()
    pixel_acc = confusion.diag().sum().item() / total if total > 0 else 0.0

    intersection = confusion.diag().float()
    union = (confusion.sum(dim = 1) + confusion.sum(dim = 0) - confusion.diag()).float()
    present = union > 0
    miou = (intersection[present] / union[present]).mean().item() if present.any() else 0.0

    print(f'Pixel Accuracy: {pixel_acc:.4f}, mIoU: {miou:.4f}')
    return pixel_acc, miou

def compute_class_weights(dataset, indices, num_classes, weight_pow, max_weight, device, cache_path = None):
    """Inverse-frequency class weights over the TRAIN split only (no val/test leakage).

    The MINC-S labels are ~54x imbalanced at the pixel level, so plain CE collapses to
    the dominant materials (mIoU stuck near 1/num_classes). `weight_pow` softens the raw
    1 / freq (0.5 = inverse-sqrt, a stable default under heavy imbalance; 0 disables it).
    Weights are normalized so present classes average to 1 (keeps the loss scale close to
    plain CE) and clamped to `max_weight` so ultra-rare classes can't dominate the gradient.
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path).to(device)

    counts = torch.zeros(num_classes, dtype = torch.double)
    for i in tqdm(indices, desc = 'Class weights'):
        label = dataset.label_only(i)
        valid = label != IGNORE_INDEX
        counts += torch.bincount(label[valid].view(-1), minlength = num_classes).double()

    present = counts > 0
    freq = counts / counts.sum().clamp(min = 1)
    weights = torch.ones(num_classes, dtype = torch.double)
    weights[present] = (1.0 / freq[present]) ** weight_pow
    weights[present] *= present.sum() / weights[present].sum()       # present-class mean -> 1
    weights = weights.clamp(max = max_weight)
    weights[~present] = 0.0                                          # classes absent from train get no weight

    weights = weights.float()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok = True)
        torch.save(weights, cache_path)
    return weights.to(device)

def main(args):
    img_size = IMG_SIZE        # 512, fixed square input required by the positional embedding
    patch_size = args.patch_size
    in_channels = 3
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    mlp_ratio = args.mlp_ratio
    drop = args.drop
    attn_drop = args.attn_drop

    num_classes = NUM_CLASSES  # 23 MINC material categories

    batch = args.batch                  # 512x512 with 4 ViT experts is heavy; raise if memory allows
    lr = args.lr
    weight_decay = args.weight_decay

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Allow TF32 on Ampere+ GPUs for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Using device: {device}')

    dataset = MINCSegmentationDataset()                     # MINC-S: 1798 labeled scenes (single split)
    n = len(dataset)
    test_size = max(1, int(0.1 * n))
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size - test_size                   # 80 / 10 / 10 train / val / test
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size = batch, shuffle = True, collate_fn = seg_collate_fn)
    val_loader = DataLoader(val_set, batch_size = batch, shuffle = False, collate_fn = seg_collate_fn)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False, collate_fn = seg_collate_fn)

    encoder = Encoder(img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop)

    self_expert_pool = nn.ModuleList([SelfExpert(img_size = embed_dim, patch_size = patch_size, in_channels = 1, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop) for _ in range(2)])
    cross_expert_pool = nn.ModuleList([CrossExpert(img_size = embed_dim, patch_size = patch_size, in_channels = 1, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop) for _ in range(2)])

    auxiliary_tokens = torch.randn(embed_dim, embed_dim)

    decoder = TransformerDecoder(embed_dim = embed_dim, num_classes = num_classes, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop)

    model = ProposedMethod(encoder, self_expert_pool, cross_expert_pool, auxiliary_tokens, decoder).to(device)

    if args.weight_pow > 0:
        cache_path = os.path.join('cache', f'clsw_n{n}_p{args.weight_pow}_m{args.max_class_weight}.pt')
        class_weights = compute_class_weights(dataset, train_set.indices, num_classes, args.weight_pow, args.max_class_weight, device, cache_path)
        print('Class weights (present):', class_weights[class_weights > 0].cpu().numpy().round(2))
    else:
        class_weights = None

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight = class_weights, ignore_index = IGNORE_INDEX)

    # Per-step warmup -> cosine decay: stabilizes the early from-scratch steps, then anneals.
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, total_iters = warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, total_steps - warmup_steps))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones = [warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, total_steps))

    for epoch in range(args.epochs):
        running_loss, num_batches = 0.0, 0
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss = train_step(model, batch_data, loss_fn, optimizer, device)
            scheduler.step()
            running_loss += loss; num_batches += 1
        print(f'Epoch: {epoch+1}, Average Loss: {running_loss / num_batches:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')

        with torch.no_grad():
            evaluate(model, val_loader, num_classes, device)

    print('Test:')
    evaluate(model, test_loader, num_classes, device)

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    
    arg.add_argument('--seed', type = int, default = 42)
    
    arg.add_argument('--batch', type = int, default = 4)
    arg.add_argument('--epochs', type = int, default = 30)
    arg.add_argument('--warmup_epochs', type = int, default = 1)

    arg.add_argument('--patch_size', type = int, default = 16)
    arg.add_argument('--embed_dim', type = int, default = 512)
    arg.add_argument('--num_heads', type = int, default = 8)
    arg.add_argument('--mlp_ratio', type = float, default = 2.0)
    arg.add_argument('--drop', type = float, default = 0.1)
    arg.add_argument('--attn_drop', type = float, default = 0.1)
    
    arg.add_argument('--lr', type = float, default = 5e-5)
    arg.add_argument('--weight_decay', type = float, default = 0.01)

    arg.add_argument('--weight_pow', type = float, default = 0.5)        # class weight = (1 / freq) ** weight_pow; 0 disables
    arg.add_argument('--max_class_weight', type = float, default = 10.0)

    args = arg.parse_args()
    main(args)
