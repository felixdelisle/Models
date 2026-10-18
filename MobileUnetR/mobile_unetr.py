import torch 
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blocks import STEM, HybridEncoder, HybridDecoder
except ImportError:  # pragma: no cover - used when running the file directly
    try:
        from blocks import STEM, HybridEncoder, HybridDecoder
    except ImportError:
        from MobileUnetR.blocks import STEM, HybridEncoder, HybridDecoder

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, channels=(32, 64, 128, 256), multiple_encoders=False):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = STEM(in_channels, c1)
        self.encoder1 = HybridEncoder(c1, c2)
        self.encoder2 = HybridEncoder(c2, c3)
        self.encoder3 = HybridEncoder(c3, c4)
        self.me = multiple_encoders 

        if multiple_encoders:  
            self.se1 = SqueezeExcitation(c1)
            self.se2 = SqueezeExcitation(c2)
            self.se3 = SqueezeExcitation(c3)
            self.se4 = SqueezeExcitation(c4)
    
    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        if self.me:
            skips = [self.se1(x1), self.se2(x2), self.se3(x3)]
            out = self.se4(x4)
        else:
            skips = [x1, x2, x3]
            out = x4
        return out, skips
    

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
    def __init__(self, in_channels=3, num_classes=1, multiple_encoders=False, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.me = multiple_encoders
        if not multiple_encoders:
            self.encoder = Encoder(in_channels, channels=channels)
        else:
            # Encoder 1 - Axial Cut
            self.encoder1 = Encoder(
                in_channels, channels=channels, multiple_encoders=True
            )
            # Encoder 2 - Coronal Cut
            self.encoder2 = Encoder(
                in_channels, channels=channels, multiple_encoders=True
            )
            # Encoder 3 - Sagittal Cut
            self.encoder3 = Encoder(
                in_channels, channels=channels, multiple_encoders=True
            )

            self.fuse_out = nn.Conv2d(3 * c4, c4, kernel_size=1)
            self.fuse_skip1 = nn.Conv2d(3 * c1, c1, kernel_size=1)
            self.fuse_skip2 = nn.Conv2d(3 * c2, c2, kernel_size=1)
            self.fuse_skip3 = nn.Conv2d(3 * c3, c3, kernel_size=1)

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
        if self.me:
            if not isinstance(x, (tuple, list)) or len(x) != 3:
                raise ValueError(
                    "multiple_encoders=True expects (axial, coronal, sagittal) input tensors"
                )
            input_size = x[0].shape[-2:]
            if any(view.shape != x[0].shape for view in x[1:]):
                raise ValueError("All three input views must have matching shapes")

            out1, skips1 = self.encoder1(x[0])
            out2, skips2 = self.encoder2(x[1])
            out3, skips3 = self.encoder3(x[2])
            out = self.fuse_out(torch.cat([out1, out2, out3], dim=1))
            skips = [
                self.fuse_skip1(torch.cat([skips1[0], skips2[0], skips3[0]], dim=1)),
                self.fuse_skip2(torch.cat([skips1[1], skips2[1], skips3[1]], dim=1)),
                self.fuse_skip3(torch.cat([skips1[2], skips2[2], skips3[2]], dim=1))
            ]
            
        else:
            if isinstance(x, (tuple, list)):
                raise ValueError(
                    "Pass one image tensor unless multiple_encoders=True"
                )
            input_size = x.shape[-2:]
            out, skips = self.encoder(x)

        x = self.decoder(out, skips)
        x = self.segmentation_head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
