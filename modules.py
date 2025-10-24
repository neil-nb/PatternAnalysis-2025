import torch.nn as nn
import torch

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p="same", groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch else ConvBNAct(in_ch, out_ch, k=1, act=False)
        self.conv1 = ConvBNAct(in_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch, act=False)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x + residual
        return self.act(x)

class sSE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))