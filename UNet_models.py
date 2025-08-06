#imports
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Projection if in_channels ≠ out_channels
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        """
        F_g: Number of channels in gating signal (from decoder ie upsampling region)
        F_l: Number of channels in skip connection (from encoder ie downsampling region)
        F_int: Number of intermediate channels (usually set to F_l // 2 or a constant like 128)
        """
        super(AttentionGate,self).__init__()
        self.W_g=nn.Sequential(
            nn.Conv2D(F_g,F_int,kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x=nn.Sequential(
            nn.Conv2D(F_l,F_int,kernel_size=1, stride=2,padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi=nn.Sequential(
            nn.Conv2D(F_int,kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x,g):
        #x: original skip connection part, g:gating signal 
        g1=self.W_g(x)
        x1=self.W_x(g)

        return x*(self.upsample(self.psi(self.relu(x1+g1))))
class AttentionUNet(nn.Module):
    def __init__(self, num_classes, block=DoubleConv):
        super(AttentionUNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        self.down_convolution_1 = block(3, 64)
        self.down_convolution_2 = block(64, 128)
        self.down_convolution_3 = block(128, 256)
        self.down_convolution_4 = block(256, 512)
        self.down_convolution_5 = block(512, 1024)

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_convolution_1 = block(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_convolution_2 = block(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_3 = block(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_4 = block(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        # Attention Gates
        self.attn1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attn2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attn3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attn4 = AttentionGate(F_g=64, F_l=64, F_int=32)

    def forward(self, x):
        # Encoder
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)

        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)

        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)

        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)

        down_9 = self.down_convolution_5(down_8)

        # Decoder
        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([self.attn1(down_7, up_1), up_1], dim=1))

        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([self.attn2(down_5, x), up_2], dim=1))

        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([self.attn3(down_3, x), up_3], dim=1))

        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([self.attn4(down_1, x), up_4], dim=1))

        return self.out(x)



class FlexibleUNet(nn.Module):
    def __init__(self, num_classes, block=DoubleConv):
        super(FlexibleUNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(2)
        
        self.down1 = block(3, 64)
        self.down2 = block(64, 128)
        self.down3 = block(128, 256)
        self.down4 = block(256, 512)
        self.down5 = block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv1 = block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = block(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.max_pool2d(d1))
        d3 = self.down3(self.max_pool2d(d2))
        d4 = self.down4(self.max_pool2d(d3))
        d5 = self.down5(self.max_pool2d(d4))

        u1 = self.conv1(torch.cat([self.up1(d5), d4], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.conv4(torch.cat([self.up4(u3), d1], dim=1))

        return self.final(u4)

class FlexibleUNetPP(nn.Module):
    def __init__(self, num_classes, block=DoubleConv):
        super(FlexibleUNetPP, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder blocks
        self.conv00 = block(3, 64)
        self.conv10 = block(64, 128)
        self.conv20 = block(128, 256)
        self.conv30 = block(256, 512)
        self.conv40 = block(512, 1024)

        # Decoder blocks (Nested)
        self.conv01 = block(64 + 128, 64)
        self.conv11 = block(128 + 256, 128)
        self.conv21 = block(256 + 512, 256)
        self.conv31 = block(512 + 1024, 512)

        self.conv02 = block(64 * 2 + 128, 64)
        self.conv12 = block(128 * 2 + 256, 128)
        self.conv22 = block(256 * 2 + 512, 256)

        self.conv03 = block(64 * 3 + 128, 64)
        self.conv13 = block(128 * 3 + 256, 128)

        self.conv04 = block(64 * 4 + 128, 64)

        self.up = lambda in_ch, out_ch: nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Upsample modules
        self.up01 = self.up(128, 64)
        self.up11 = self.up(256, 128)
        self.up21 = self.up(512, 256)
        self.up31 = self.up(1024, 512)

        self.up02 = self.up(128, 64)
        self.up12 = self.up(256, 128)
        self.up22 = self.up(512, 256)

        self.up03 = self.up(128, 64)
        self.up13 = self.up(256, 128)

        self.up04 = self.up(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))

        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up21(x30)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up31(x40)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up12(x21)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up22(x31)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up03(x12)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up13(x22)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up04(x13)], dim=1))

        return self.final(x04)
    