import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import STEM, HybridEncoder, HybridDecoder
except ImportError:  # pragma: no cover - used when running the file directly
    try:
        from blocks import STEM, HybridEncoder, HybridDecoder
    except ImportError:
        from MobileUnetR.blocks import STEM, HybridEncoder, HybridDecoder

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = STEM(in_channels, c1)
        self.encoder1 = HybridEncoder(c1, c2)
        self.encoder2 = HybridEncoder(c2, c3)
        self.encoder3 = HybridEncoder(c3, c4)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        skips = [x1, x2, x3]
        return x4, skips

class Decoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.decoder1 = HybridDecoder(c4, c3, c3)
        self.decoder2 = HybridDecoder(c3, c2, c2)
        self.decoder3 = HybridDecoder(c2, c1, c1)

    def forward(self, x, skips):
        x = self.decoder1(x, skips[2])
        x = self.decoder2(x, skips[1])
        x = self.decoder3(x, skips[0])
        return x

class MobileUNETR(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder = Encoder(in_channels, channels)
        self.decoder = Decoder(channels)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )


    def forward(self, x):
        input_size = x.shape[-2:]
        bottleneck, skips = self.encoder(x)
        x = self.decoder(bottleneck, skips)
        x = self.segmentation_head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


