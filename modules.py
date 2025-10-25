import torch.nn as nn
import torch
import torch.nn.functional as F

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
    
class cSE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, kernel_size=1)
        self.fc2 = nn.Conv2d(ch // r, ch, kernel_size=1)

    def forward(self, x):
        z = F.adaptive_avg_pool2d(x, 1)
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))
        return x * z
    
class scSE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.s = sSE(ch)
        self.c = cSE(ch)
        
    def forward(self, x):
        return self.s(x) + self.c(x)

class AttentionGate(nn.Module):
    def __init__(self, in_ch_x, in_ch_g, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(inter_ch)

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = F.relu(self.bn(theta_x + phi_g))
        att = torch.sigmoid(self.psi(f))
        att = F.interpolate(att, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * att
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout_p=0.0, use_attn=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(skip_ch, out_ch, inter_ch=out_ch // 2) if use_attn else None
        self.fuse = ResidualBlock(out_ch + skip_ch, out_ch, dropout_p=dropout_p)
        self.scse = scSE(out_ch)
        
    def forward(self, x, skip):
        x = self.up(x)
        if self.attn:
            skip = self.attn(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.scse(self.fuse(x))
