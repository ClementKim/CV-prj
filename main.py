import torch
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

def main():
    img_size = IMG_SIZE        # 512, fixed square input required by the positional embedding
    patch_size = 16
    in_channels = 3
    embed_dim = 512
    num_heads = 8
    mlp_ratio = 4.0
    drop = 0.1
    attn_drop = 0.1

    num_classes = NUM_CLASSES  # 23 MINC material categories

    batch = 2                  # 512x512 with 4 ViT experts is heavy; raise if memory allows
    lr = 5e-5
    weight_decay = 0.01

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

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index = IGNORE_INDEX)

    for epoch in range(5):
        running_loss, num_batches = 0.0, 0
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss = train_step(model, batch_data, loss_fn, optimizer, device)
            running_loss += loss; num_batches += 1
        print(f'Epoch: {epoch}, Average Loss: {running_loss / num_batches:.4f}')

        with torch.no_grad():
            evaluate(model, val_loader, num_classes, device)

    print('Test:')
    evaluate(model, test_loader, num_classes, device)

if __name__ == '__main__':
    main()