import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from encoder import TransformerEncoder, ResNetEncoder
from transformer import MLP, SelfExpert, CrossExpert

class Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token = True):
        super(Encoder, self).__init__()

        self.transformer_encoder = TransformerEncoder(img_size = img_size, patch_size = patch_size, in_channels = in_channels, embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, drop = drop, attn_drop = attn_drop, is_cls_token = is_cls_token)
        self.resnet_encoder = ResNetEncoder(in_channels = in_channels, out_channels = embed_dim)

    def forward(self, x):
        return self.transformer_encoder(x), self.resnet_encoder(x)
    
class DenseMoE(nn.Module):
    def __init__(self, self_expert_pool, cross_expert_pool):
        super(DenseMoE, self).__init__()

        self.self_expert_pool = self_expert_pool
        self.cross_expert_pool = cross_expert_pool

    def forward(self, bilinear_feature):
        self_expert_output = [expert(bilinear_feature) for expert in self.self_expert_pool]
        cross_expert_output = [expert(bilinear_feature) for expert in self.cross_expert_pool]

        aggregated_expert_output = (sum(self_expert_output) + sum(cross_expert_output)).mean(dim = 0)

        return aggregated_expert_output
    
class ProposedMethod(nn.Module):
    def __init__(self, encoder, self_expert_pool, cross_expert_pool):
        super(ProposedMethod, self).__init__()

        self.encoder = encoder
        self.moe = DenseMoE(self_expert_pool, cross_expert_pool)

    def forward(self, x):
        transformer_feature, resnet_feature = self.encoder(x)

        bilinear_feature = torch.bmm(torch.unsqueeze(transformer_feature, dim = 2), torch.unsqueeze(resnet_feature, dim = 1)).view(x.size(0), -1)

        moe_output = self.moe(bilinear_feature)

        return moe_output
    
def train_step(model, batch, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()

    images, labels = batch

    loss = loss_fn(model(images), labels)

    loss.backward()
    optimizer.step()

    return loss.item()
    
def main():
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    drop = 0.1
    attn_drop = 0.1

    batch = 128
    lr = 5e-5
    weight_decay = 0.01

    train_loader = DataLoader(
        CIFAR10('data', train = True, download = True),
        batch_size = batch,
        shuffle = True
    )

    test_loader = DataLoader(
        CIFAR10('data', train = False, download = True),
        batch_size = batch,
        shuffle = False
    )

    encoder = Encoder(img_size, patch_size, in_channels, embed_dim, num_heads, mlp_ratio, drop, attn_drop)
    x = torch.randn(1, in_channels, img_size, img_size)
    transformer_feature, resnet_feature = encoder(x)

    bilinear_feature = torch.bmm(torch.unsqueeze(transformer_feature, dim = 2), torch.unsqueeze(resnet_feature, dim = 1)).view(1, -1)

    self_expert_pool = nn.ModuleList([SelfExpert(embed_dim * embed_dim, mlp_ratio, drop) for _ in range(2)])
    cross_expert_pool = nn.ModuleList([CrossExpert(embed_dim * embed_dim, mlp_ratio, drop) for _ in range(2)])

    model = ProposedMethod(encoder, self_expert_pool, cross_expert_pool)

    parameters = {
        'transformer_encoder': encoder.transformer_encoder.parameters(),
        'resnet_encoder': encoder.resnet_encoder.parameters(),
        'self_experts': [param for expert in self_expert_pool for param in expert.parameters()],
        'cross_experts': [param for expert in cross_expert_pool for param in expert.parameters()]
    }

    optimizer = torch.optim.AdamW(parameters, lr = lr, weight_decay = weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        for batch in train_loader:
            loss = train_step(model, batch, loss_fn, optimizer)

            print(f'Epoch: {epoch}, Loss: {loss}')

    with torch.no_grad():
        model.eval()

        total_correct = 0
        total_samples = 0

        for batch in test_loader:
            images, labels = batch

            outputs = model(images)
            _, predicted = torch.max(outputs, dim = 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Test Accuracy: {accuracy:.4f}')
