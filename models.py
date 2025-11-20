from configuration import config
from torchvision.models import vgg16, VGG16_Weights, efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights
from torch import nn
import torch

# 注意力机制
class CBAM(nn.Module):
    # 通道注意力和空间注意力的混合注意力机制
    def __init__(self, channels, reduction=16): # reduction，通道压缩比例，可减少参数量
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 降维
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(),
            # 升维
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            # 输入是2个通道（平均池化和最大池化的结果）
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    # 输入形状应为[batch_size, channels, height, width]
    def forward(self, x):

        # 通道注意力
        # 计算通道注意力权重并应用到特征图上
        ca_weights = self.channel_attention(x)
        x_with_ca = x * ca_weights

        # 空间注意力
        avg_pool = torch.mean(x_with_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_with_ca, dim=1, keepdim=True)
        # 拼接平均池化和最大池化的结果，形状为 [batch_size, 2, height, width]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weights = self.spatial_attention(spatial_input)
        # 将空间注意力权重应用到特征图上
        x_with_both_attentions = x_with_ca * sa_weights

        # 返回经过注意力加权的特征图
        return x_with_both_attentions


class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=len(config.classNames), freeze_backbone=True):
        super().__init__()

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 只使用efficientnet的特征提取部分来提取特征，不用原始的分类器
        self.features = self.backbone.features
        
        # 部分冻结策略：优先泛化能力
        if freeze_backbone:
            # 冻结前65%的层（解冻后35%），优先保证泛化
            num_layers = len(self.features)
            freeze_until = int(num_layers * 0.65)  # 冻结前65%
            for i, layer in enumerate(self.features):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
                # 后35%可训练，减少过拟合风险


        # 在特征提取后添加CBAM注意力机制
        # 让模型学会关注睑板腺图像中的重要区域
        self.cbam = CBAM(1280, reduction=16)

        # 将输入图片池化为固定大小，这样不受输入图片尺寸影响
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 自定义分类器,替换原始的分类器，设计适合睑板腺分类任务的新分类器
        # 高Dropout，强力防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.62),  # 高dropout
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.52),  # 高dropout
            nn.Linear(512, num_classes)
        )

        # 移除原始分类器
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        features = self.features(x)
        attended_features = self.cbam(features)
        pooled = self.avgpool(attended_features)
        pooled = torch.flatten(pooled, 1)
        output = self.classifier(pooled)

        return output


