import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p="same", groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=p, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
