import torch 
import torch.nn as nn
import torch.nn.functional as F

class MobileViTBlock(nn.Module):

    def __init__(self, channels, transformer_dim=64, depth=2, patch_size=2,
                 heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size

        # Représentation locale
        self.local_rep = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.SiLU(inplace=True),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.transformer_norm = nn.LayerNorm(transformer_dim)

        # Projection retour + fusion
        self.proj_out = nn.Conv2d(transformer_dim, channels, kernel_size=1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def _unfold(self, x):
        B, d, H, W = x.shape
        p = self.patch_size
        # x: (B, d, H, W) -> patches de taille p x p
        x = x.reshape(B, d, H // p, p, W // p, p)
        x = x.permute(0, 3, 5, 1, 2, 4)  # B, p, p, d, H//p, W//p
        num_patches = (H // p) * (W // p)
        x = x.reshape(B * p * p, d, num_patches)
        x = x.permute(0, 2, 1)  # (B*p*p, num_patches, d)
        return x, (B, d, H, W)

    def _fold(self, x, shape):
        B, d, H, W = shape
        p = self.patch_size
        x = x.permute(0, 2, 1)  # (B*p*p, d, num_patches)
        x = x.reshape(B, p, p, d, H // p, W // p)
        x = x.permute(0, 3, 4, 1, 5, 2)  # B, d, H//p, p, W//p, p
        x = x.reshape(B, d, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # padding si H/W non divisibles par patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        x_in = x
        if pad_h or pad_w:
            x_in = F.pad(x_in, (0, pad_w, 0, pad_h))

        y = self.local_rep(x_in)
        y, shape = self._unfold(y)
        y = self.transformer_norm(self.transformer(y))
        y = self._fold(y, shape)
        y = self.proj_out(y)

        if pad_h or pad_w:
            y = y[:, :, :H, :W]

        out = self.fusion(torch.cat([x, y], dim=1))
        return out
    
class MobileNetv2(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio = 4, stride = 1 ):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ]
    
        self.block = nn.Sequential(*layers)

        if not self.use_residual:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        if x.shape[-2:] != out.shape[-2:]:
            x = F.interpolate(x, size=out.shape[-2:], mode="bilinear", align_corners=False)
        if x.shape[1] != out.shape[1]:
            x = self.proj(x)
        return out + x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class STEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.mn1 = MobileNetv2(out_channels, out_channels, stride=1)
        self.mn2 = MobileNetv2(out_channels, out_channels, stride=2)
        self.mn3 = MobileNetv2(out_channels, out_channels, stride=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.mn1(x)
        x = self.mn2(x)
        x = self.mn3(x)
        return x


class HybridEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, transformer_dim=64, depth=2):
        super().__init__()
        self.downsample = MobileNetv2(in_channels, out_channels, stride=2)
        self.refine = MobileNetv2(out_channels, out_channels, stride=1)
        self.mobilevit = MobileViTBlock(out_channels, transformer_dim=transformer_dim, depth=depth)

    def forward(self, x):
        x = self.downsample(x)
        x = self.refine(x)
        x = self.mobilevit(x)
        return x
    
class HybridDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, transformer_dim=64, depth=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.local_refine = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
        self.mobilevit = MobileViTBlock(out_channels, transformer_dim=transformer_dim, depth=depth)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.local_refine(torch.cat([x, skip], dim=1))
        x = self.mobilevit(x)
        return x


    