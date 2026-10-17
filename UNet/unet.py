import torch 
import torch.nn as nn
import torch.nn.functional as F
from blocks import EncoderBlock, DecoderBlock

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder1 = EncoderBlock(in_channels, c1)
        self.encoder2 = EncoderBlock(c1, c2)
        self.encoder3 = EncoderBlock(c2, c3)
        self.encoder4 = EncoderBlock(c3, c4)

    def forward(self, x):
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, _ = self.encoder4(x3)
        skips = [skip1, skip2, skip3]
        return x4, skips

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
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, channels=(32, 64, 128, 256)):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, channels)
        self.decoder = Decoder(base_channels, channels)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x4, skips = self.encoder(x)
        x = self.decoder(x4, skips)
        x = self.final_conv(x)
        return x