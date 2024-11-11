import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 10


# 定义Auto-Encoder结构
# 就这么简单
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # 将输入压缩到3维（低维表示）
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),  # 还原到输入的原始尺寸
            nn.Sigmoid()  # 使用Sigmoid将输出限定在[0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def train(model, train_loader, criterion, optimizer, first_epochs):
    # 训练过程
    for epoch in range(first_epochs):
        for data, _ in train_loader:
            # 前向传播
            output = model(data)
            loss = criterion(output, data)  # 计算重构误差

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{first_epochs}], Loss: {loss.item():.4f}')


def evaluate_and_print_result(model, train_loader):
    examples = enumerate(train_loader)
    batch_idx, (example_data, _) = next(examples)

    with torch.no_grad():
        output = model(example_data)

    # 显示原始图像和重构图像
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(example_data[i].view(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(output[i].view(28, 28), cmap='gray')
        axes[1, i].set_title("Reconstructed")

    for ax in axes.flatten():
        ax.axis('off')
    plt.show()


def show_latent_space(model, train_loader):
    model.eval()  # 设置模型为评估模式
    latents = []
    labels = []

    with torch.no_grad():
        for data, target in train_loader:
            encoded = model.encoder(data)  # 获取编码器输出，即隐空间表示
            latents.append(encoded)
            labels.append(target)

    # 将数据拼接成张量
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # 可视化latent space（3D图）
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latents[:, 0].numpy(), latents[:, 1].numpy(), latents[:, 2].numpy(), c=labels.numpy(),
                         cmap='viridis', alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Digits")
    ax.add_artist(legend)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    plt.show()


# export all methods

__all__ = ['train', 'evaluate_and_print_result', 'show_latent_space']
