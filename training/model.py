"""可配置残差 CNN（双头 policy+value），供五子棋监督训练与后续自对弈使用。

配置:  --blocks/--channels 任意调整 (4x64 ~ 12x160)
默认:   8 blocks x 128 channels (~2.4M 参数)
输入:   (B, 2, 15, 15)  通道0=己方, 通道1=对方
输出:   policy logits (B, 225), value logits (B, 3)   (胜/负/和)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity)


class GomokuNet(nn.Module):
    def __init__(self, board: int = 15, in_channels: int = 2,
                 channels: int = 128, blocks: int = 8,
                 policy_out: int = 225, value_out: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 1, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, value_out),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        p = self.policy_conv(x).flatten(1)           # (B, 225)
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)   # 全局平均池化
        v = self.value_fc(g)                         # (B, 3)
        return p, v

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
