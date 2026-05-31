import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10

from encoder import TransformerEncoder, ResNetEncoder
from transformer import SelfExpert, CrossExpert

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
    def __init__(self, encoder, self_expert_pool, cross_expert_pool, aux, embed_dim, num_classes):
        super(ProposedMethod, self).__init__()

        self.encoder = encoder
        self.moe = DenseMoE(self_expert_pool, cross_expert_pool, aux)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        transformer_feature, resnet_feature = self.encoder(x)

        bilinear_feature = torch.bmm(transformer_feature.unsqueeze(2), resnet_feature.unsqueeze(1)).unsqueeze(1) # (B, 1, embed_dim, embed_dim)

        moe_output = self.moe(bilinear_feature)

        return self.head(moe_output)

def train_step(model, batch, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()

    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    loss = loss_fn(model(images), labels)

    loss.backward()
    optimizer.step()

    return loss.item()
    
def main():
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 512
    num_heads = 8
    mlp_ratio = 4.0
    drop = 0.1
    attn_drop = 0.1

    num_classes = 10 # for CIFAR-10

    batch = 128
    lr = 5e-5
    weight_decay = 0.01

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_loader = DataLoader(
        CIFAR10('data', train = True, download = True, transform = transform),
        batch_size = batch,
        shuffle = True
    )

    test_loader = DataLoader(
        CIFAR10('data', train = False, download = True, transform = transform),
        batch_size = batch,
        shuffle = False
    )

    encoder = Encoder(img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop)

    self_expert_pool = nn.ModuleList([SelfExpert(img_size = embed_dim, patch_size = patch_size, in_channels = 1, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop) for _ in range(2)])
    cross_expert_pool = nn.ModuleList([CrossExpert(img_size = embed_dim, patch_size = patch_size, in_channels = 1, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop) for _ in range(2)])

    auxiliary_tokens = torch.randn(embed_dim, embed_dim)

    model = ProposedMethod(encoder, self_expert_pool, cross_expert_pool, auxiliary_tokens, embed_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        running_loss, num_batches = 0.0, 0
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss = train_step(model, batch_data, loss_fn, optimizer, device)
            running_loss += loss; num_batches += 1
        print(f'Epoch: {epoch}, Average Loss: {running_loss / num_batches:.4f}')


    with torch.no_grad():
        model.eval()

        total_correct = 0
        total_samples = 0

        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim = 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()