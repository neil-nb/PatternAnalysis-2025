import torch.nn as nn
import torch
import torch.nn.functional as F

# Building Blocks
class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p="same", groups=1, act=True):
        """
        Convolution + BatchNorm + Activation block.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (str or int): Padding.
            groups (int): Number of groups for grouped convolution.
            act (bool): Whether to apply ReLU activation.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        """Forward pass through the Conv-BN-Act block."""
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Two 3x3 convs with BN+ReLU and a residual shortcut."""
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        """
        Residual block with two convolutional layers.
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch else ConvBNAct(in_ch, out_ch, k=1, act=False)
        self.conv1 = ConvBNAct(in_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch, act=False)
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the residual block."""
        residual = self.proj(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x + residual
        return self.act(x)

class sSE(nn.Module):
    """Spatial Squeeze and Excitation"""
    def __init__(self, ch):
        """
        Spatial Squeeze and Excitation block.
        Args:
            ch (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        """Forward pass through the Spatial Squeeze and Excitation block."""
        return x * torch.sigmoid(self.conv(x))
    
class cSE(nn.Module):
    """Channel Squeeze and Excitation"""
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, kernel_size=1)
        self.fc2 = nn.Conv2d(ch // r, ch, kernel_size=1)

    def forward(self, x):
        """Forward pass through the Channel Squeeze and Excitation block."""
        z = F.adaptive_avg_pool2d(x, 1)
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))
        return x * z
    
class scSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze and Excitation"""
    def __init__(self, ch):
        super().__init__()
        self.s = sSE(ch)
        self.c = cSE(ch)
        
    def forward(self, x):
        """Forward pass through the Concurrent Spatial and Channel Squeeze and Excitation block."""
        return self.s(x) + self.c(x)

class AttentionGate(nn.Module):
    """Additive attention on skip connections (Attention U-Net style)."""
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
    """Upsampling block with optional attention and scSE."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout_p=0.0, use_attn=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(skip_ch, out_ch, inter_ch=out_ch // 2) if use_attn else None
        self.fuse = ResidualBlock(out_ch + skip_ch, out_ch, dropout_p=dropout_p)
        self.scse = scSE(out_ch)

    def forward(self, x, skip):
        """Forward pass through the upsampling block."""
        x = self.up(x)
        if self.attn:
            skip = self.attn(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.scse(self.fuse(x))

# Main Model
class ImprovedUNet(nn.Module):
    """Improved U-Net with residual connections, scSE attention, and deep supervision."""
    def __init__(self, num_classes, base_ch=64, dropout_p=0.1, deep_supervision=True):
        """Initialize the Improved U-Net model.
        Args:
            num_classes (int): Number of output segmentation classes.
            base_ch (int): Base number of channels for the model.
            dropout_p (float): Dropout probability.
            deep_supervision (bool): Whether to use deep supervision.
        """
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder
        self.e1 = nn.Sequential(ResidualBlock(1, base_ch, dropout_p), scSE(base_ch))
        self.p1 = nn.MaxPool2d(2)

        self.e2 = nn.Sequential(ResidualBlock(base_ch, base_ch*2, dropout_p), scSE(base_ch*2))
        self.p2 = nn.MaxPool2d(2)

        self.e3 = nn.Sequential(ResidualBlock(base_ch*2, base_ch*4, dropout_p), scSE(base_ch*4))
        self.p3 = nn.MaxPool2d(2)

        self.e4 = nn.Sequential(ResidualBlock(base_ch*4, base_ch*8, dropout_p), scSE(base_ch*8))
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.b = nn.Sequential(ResidualBlock(base_ch*8, base_ch*16, dropout_p), scSE(base_ch*16))

        # Decoder
        self.d4 = UpSampleBlock(base_ch*16, base_ch*8, base_ch*8, dropout_p)
        self.d3 = UpSampleBlock(base_ch*8, base_ch*4, base_ch*4, dropout_p)
        self.d2 = UpSampleBlock(base_ch*4, base_ch*2, base_ch*2, dropout_p)
        self.d1 = UpSampleBlock(base_ch*2, base_ch, base_ch, dropout_p)

        # Output layers
        self.out = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(base_ch*2, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(base_ch*4, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass through the Improved U-Net model."""
        # Encoder
        e1 = self.e1(x); p1 = self.p1(e1)
        e2 = self.e2(p1); p2 = self.p2(e2)
        e3 = self.e3(p2); p3 = self.p3(e3)
        e4 = self.e4(p3); p4 = self.p4(e4)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)

        out = self.out(d1)

        # Deep Supervision
        if self.deep_supervision and self.training:
            aux2 = F.interpolate(self.aux2(d2), size=out.shape[2:], mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.aux3(d3), size=out.shape[2:], mode="bilinear", align_corners=False)
            return out, aux2, aux3
        return out
