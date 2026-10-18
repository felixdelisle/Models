import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .blocks import EncoderBlock, DecoderBlock
except ImportError:
    from blocks import EncoderBlock, DecoderBlock


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
    def __init__(self, in_channels=3, base_channels=32, channels=(32, 64, 128, 256), multiple_encoders=False,):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder1 = EncoderBlock(in_channels, c1)
        self.encoder2 = EncoderBlock(c1, c2)
        self.encoder3 = EncoderBlock(c2, c3)
        self.encoder4 = EncoderBlock(c3, c4)
        self.me = multiple_encoders

        if multiple_encoders:
            self.se1 = SqueezeExcitation(c1)
            self.se2 = SqueezeExcitation(c2)
            self.se3 = SqueezeExcitation(c3)
            self.se4 = SqueezeExcitation(c4)

    def forward(self, x):
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, _ = self.encoder4(x3)

        if self.me:
            skips = [self.se1(skip1), self.se2(skip2), self.se3(skip3)]
            out = self.se4(x4)
        else:
            skips = [skip1, skip2, skip3]
            out = x4
        return out, skips

class Decoder(nn.Module):
    def __init__(self, base_channels=32, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.decoder1 = DecoderBlock(c4, c3, c3)
        self.decoder2 = DecoderBlock(c3, c2, c2)
        self.decoder3 = DecoderBlock(c2, c1, c1)

    def forward(self, x, skips):
        skip1, skip2, skip3 = skips
        x = self.decoder1(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder3(x, skip1)
        return x

class AttentionUNet(nn.Module):
    def __init__( self, in_channels=3, out_channels=1, base_channels=32, channels=(32, 64, 128, 256), multiple_encoders=False):
        super().__init__()
        self.me = multiple_encoders
        if multiple_encoders:
            c1, c2, c3, c4 = channels
            self.encoder1 = Encoder(in_channels, base_channels, channels, multiple_encoders=True)
            self.encoder2 = Encoder(in_channels, base_channels, channels, multiple_encoders=True)
            self.encoder3 = Encoder(in_channels, base_channels, channels, multiple_encoders=True)
            self.fuse_out = nn.Conv2d(3 * c4, c4, kernel_size=1)
            self.fuse_skip1 = nn.Conv2d(3 * c1, c1, kernel_size=1)
            self.fuse_skip2 = nn.Conv2d(3 * c2, c2, kernel_size=1)
            self.fuse_skip3 = nn.Conv2d(3 * c3, c3, kernel_size=1) 
        else:
            self.encoder = Encoder(in_channels, base_channels, channels)
        self.decoder = Decoder(base_channels, channels)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        if self.me:
            out1, skips1 = self.encoder1(x[0])
            out2, skips2 = self.encoder2(x[1])
            out3, skips3 = self.encoder3(x[2])
            x4 = self.fuse_out(torch.cat([out1, out2, out3], dim=1))
            skips = [
                self.fuse_skip1(
                    torch.cat([skips1[0], skips2[0], skips3[0]], dim=1)
                ),
                self.fuse_skip2(
                    torch.cat([skips1[1], skips2[1], skips3[1]], dim=1)
                ),
                self.fuse_skip3(
                    torch.cat([skips1[2], skips2[2], skips3[2]], dim=1)
                ),
            ]
        else:
            x4, skips = self.encoder(x)

        x = self.decoder(x4, skips)
        x = self.final_conv(x)
        return x