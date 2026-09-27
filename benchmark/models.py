"""
Model architectures for the ESC-50 benchmark.

  PiczakCNN   - reimplementation (adapted to the 128-mel protocol) of the
                classic piczak2015b baseline (published 64.5%). Used as a
                controlled, apples-to-apple reference trained under the exact
                same pipeline as the new model.

  BCResSANet  - the NEW proposed architecture:
                *Broadcasting-Residual + Squeeze-Excitation + Self-Attentive*
                network. A parameter-efficient residual CNN whose blocks
                factorize processing into a 2-D frequency path and a broadcast
                1-D temporal path, recalibrated by squeeze-excitation channel
                attention, and pooled by a self-attentive temporal head instead
                of global average pooling. Designed to generalize from only
                1600 clips/fold and to fit an 8 GB GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Baseline: Piczak 2015 CNN (adapted)
# --------------------------------------------------------------------------
class PiczakCNN(nn.Module):
    def __init__(self, n_classes=50, in_ch=1, p_drop=0.5):
        super().__init__()
        # tall "vertical" filter spanning much of the frequency axis (Piczak's idea)
        self.conv1 = nn.Conv2d(in_ch, 80, kernel_size=(57, 6), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3))
        self.conv2 = nn.Conv2d(80, 80, kernel_size=(1, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.gap = nn.AdaptiveAvgPool2d((4, 4))  # keeps FC small on full-length input
        self.drop = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(80 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# --------------------------------------------------------------------------
# Building blocks for the new model
# --------------------------------------------------------------------------
class SqueezeExcite(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 4), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 4), ch, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BCResBlock(nn.Module):
    """
    Broadcasting-Residual block (Kim et al. 2021, adapted) + Squeeze-Excitation.

    Two complementary paths added to a residual:
      * 2-D frequency path : depthwise 3x1 conv over frequency (freq-local)
      * 1-D temporal path  : the freq-averaged signal goes through a depthwise
                             1x3 temporal conv, then is BROADCAST back over all
                             frequency bins.
    A pointwise mixing conv + SE recalibrates channels. This factorization is
    very parameter-efficient, which matters with only 1600 training clips.
    """
    def __init__(self, in_ch, out_ch, stride=(1, 1), drop=0.1):
        super().__init__()
        self.same = (in_ch == out_ch) and (stride == (1, 1))
        # pointwise expansion / projection
        self.pw_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        # frequency-local depthwise conv (3 over freq, 1 over time)
        self.freq_dw = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (3, 1), stride=stride, padding=(1, 0),
                      groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # temporal depthwise conv (1 over freq, 3 over time) on freq-averaged feat.
        # Time is already downsampled by freq_dw's stride, so this stays stride 1.
        self.temp_dw = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1,
                      groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(out_ch)
        self.pw_out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = nn.Dropout2d(drop)
        if not self.same:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        idt = self.proj(x)
        h = self.pw_in(x)
        # frequency path
        f = self.freq_dw(h)                       # [B, C, F', T']
        # temporal path: average over frequency -> [B, C, T'] -> conv -> broadcast
        t = f.mean(dim=2)                          # [B, C, T']
        t = self.temp_dw(t)                        # [B, C, T']
        t = t.unsqueeze(2)                         # [B, C, 1, T']
        h = F.relu(f + t)                          # broadcast add
        h = self.se(h)
        h = self.pw_out(h)
        h = self.drop(h)
        return F.relu(h + idt)


class AttentivePool(nn.Module):
    """Self-attentive statistics pooling over the time axis."""
    def __init__(self, ch, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(ch, hidden, 1), nn.Tanh(),
            nn.Conv1d(hidden, ch, 1),
        )

    def forward(self, x):                          # x: [B, C, T]
        w = torch.softmax(self.attn(x), dim=2)     # per-channel attention over time
        mean = torch.sum(w * x, dim=2)
        var = torch.sum(w * (x - mean.unsqueeze(2)) ** 2, dim=2).clamp(min=1e-6)
        return torch.cat([mean, torch.sqrt(var)], dim=1)   # [B, 2C]


class BCResSANet(nn.Module):
    """The proposed architecture. Input: [B, 1, 128, 431] log-mel."""
    def __init__(self, n_classes=50, width=32, drop=0.15):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8   # 32/64/128/256
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),                                  # 128x431 -> 64x215
        )
        self.stage1 = nn.Sequential(
            BCResBlock(c1, c2, stride=(2, 2), drop=drop),          # 64x215 -> 32x108
            BCResBlock(c2, c2, drop=drop),
        )
        self.stage2 = nn.Sequential(
            BCResBlock(c2, c3, stride=(2, 2), drop=drop),          # -> 16x54
            BCResBlock(c3, c3, drop=drop),
        )
        self.stage3 = nn.Sequential(
            BCResBlock(c3, c4, stride=(2, 2), drop=drop),          # -> 8x27
            BCResBlock(c4, c4, drop=drop),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))           # collapse frequency
        self.attn_pool = AttentivePool(c4)
        self.head = nn.Sequential(
            nn.Dropout(drop), nn.Linear(2 * c4, c4), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(c4, n_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.freq_pool(x)          # [B, C, 1, T]
        x = x.squeeze(2)               # [B, C, T]
        x = self.attn_pool(x)          # [B, 2C]
        return self.head(x)


# --------------------------------------------------------------------------
# Proposed model: FRESCA-Net
#   Frequency-aware Residual-SE net + Energy-aware BC + multi-Crop + Attentive
#   statistics pooling. SE-ResNet-18 backbone (~2.9M params) with anisotropic
#   downsampling (frequency /16, time /4) so a real temporal sequence survives
#   for the attentive-statistics temporal pooling head. Input is 2-channel:
#   the standardized log-mel + a fixed frequency-coordinate map (CoordConv),
#   because frequency is not translation-invariant on a mel axis.
# --------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth on the residual branch."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep)
        return x / keep * mask


class SEBasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), r=8, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SqueezeExcite(cout, r=r)
        self.drop_path = DropPath(drop_path)
        if cin != cout or stride != (1, 1):
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        idt = self.short(x)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(self.drop_path(h) + idt, inplace=True)


class AttentiveStatsPool(nn.Module):
    """Single-head attentive statistics pooling over time -> [mean; std]."""
    def __init__(self, ch, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(ch, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, x):                          # x: [B, C, L]
        a = torch.softmax(self.score(x), dim=2)    # [B, 1, L]
        mu = torch.sum(a * x, dim=2)               # [B, C]
        var = torch.sum(a * x * x, dim=2) - mu * mu
        sigma = var.clamp(min=1e-5).sqrt()
        return torch.cat([mu, sigma], dim=1)       # [B, 2C]


class FRESCANet(nn.Module):
    def __init__(self, n_classes=50, in_ch=2, widths=(32, 64, 128, 256),
                 max_droppath=0.10, head_drop=0.5):
        super().__init__()
        c1, c2, c3, c4 = widths
        dp = [0.0, max_droppath / 3, 2 * max_droppath / 3, max_droppath]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.stage1 = nn.Sequential(
            SEBasicBlock(c1, c1, (1, 1), r=8, drop_path=dp[0]),
            SEBasicBlock(c1, c1, (1, 1), r=8, drop_path=dp[0]),
        )
        self.stage2 = nn.Sequential(
            SEBasicBlock(c1, c2, (2, 2), r=8, drop_path=dp[1]),      # freq/2 time/2
            SEBasicBlock(c2, c2, (1, 1), r=8, drop_path=dp[1]),
        )
        self.stage3 = nn.Sequential(
            SEBasicBlock(c2, c3, (2, 1), r=16, drop_path=dp[2]),     # freq/2 time kept
            SEBasicBlock(c3, c3, (1, 1), r=16, drop_path=dp[2]),
        )
        self.stage4 = nn.Sequential(
            SEBasicBlock(c3, c4, (2, 1), r=16, drop_path=dp[3]),     # freq/2 time kept
            SEBasicBlock(c4, c4, (1, 1), r=16, drop_path=dp[3]),
        )
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))
        self.pool = AttentiveStatsPool(c4)
        self.head = nn.Sequential(
            nn.BatchNorm1d(2 * c4), nn.Dropout(head_drop), nn.Linear(2 * c4, n_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                          # x: [B, 2, M, T]
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.freq_collapse(x).squeeze(2)       # [B, C, L]
        x = self.pool(x)                           # [B, 2C]
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODELS = {"piczak": PiczakCNN, "bcressa": BCResSANet, "fresca": FRESCANet}


if __name__ == "__main__":
    for name, cls in MODELS.items():
        m = cls()
        in_ch = 2 if name == "fresca" else 1
        for T in (258, 431):                       # 3s train crop and full 5s
            x = torch.randn(4, in_ch, 128, T)
            y = m(x)
            print(f"{name:10s} T={T} params={count_params(m)/1e6:.2f}M out={tuple(y.shape)}")
