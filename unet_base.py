# Directory structure:
# ├── model.py
# ├── train.py
# ├── dataset.py
# ├── metrics.py
# ├── utils.py
# └── requirements.txt

# model.py
import torch
import torch.nn as nn
    
def center_crop(enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
    """
    Crop encoder feature map to the size of decoder feature map (center crop).
    enc_feat: [N, C_enc, H_e, W_e]
    dec_feat: [N, C_dec, H_d, W_d]
    returns: [N, C_enc, H_d, W_d]
    """
    _, _, H_e, W_e = enc_feat.size()
    _, _, H_d, W_d = dec_feat.size()
    delta_h = (H_e - H_d) // 2
    delta_w = (W_e - W_d) // 2
    return enc_feat[:, :, delta_h:delta_h + H_d, delta_w:delta_w + W_d]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)



    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # -> [N,64,H-4,W-4]
        x2 = self.down1(x1)    # -> [N,128,(H-4)/2-4,(W-4)/2-4]
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5)                       # upsample
        x4_c = center_crop(x4, x)             # crop encoder feature
        x = torch.cat([x, x4_c], dim=1)       # concat
        x = self.conv1(x)

        x = self.up2(x)
        x3_c = center_crop(x3, x)
        x = torch.cat([x, x3_c], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x2_c = center_crop(x2, x)
        x = torch.cat([x, x2_c], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x1_c = center_crop(x1, x)
        x = torch.cat([x, x1_c], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits
