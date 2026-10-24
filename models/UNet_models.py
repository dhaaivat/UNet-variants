#imports
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_input_size(x, factor=16):
    # Four 2x2 poolings: H and W must be divisible by 16 for skips to line up.
    if x.shape[-2] % factor or x.shape[-1] % factor:
        raise ValueError(f"Input H and W must be divisible by {factor}, got {tuple(x.shape[-2:])}")


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2, as in Ronneberger et al. 2015 with two deliberate deviations:
    padding=1 keeps H and W unchanged (the paper uses unpadded convs and crops the skips), and
    BatchNorm is added (not in the paper, but standard practice and used by the Attention U-Net code).
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """Basic residual block (He et al. 2016): conv-BN-ReLU-conv-BN, add shortcut, ReLU."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection if in_channels ≠ out_channels
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class AttentionGate(nn.Module):
    """Additive attention gate, Oktay et al. 2018, eqs. 1-2:
        alpha = sigmoid(psi(relu(W_x x + W_g g + b_g)) + b_psi),   output = x * alpha
    The gating signal g comes from one level coarser than x, so x is downsampled to g's
    resolution (2x2 conv, stride 2, as in the authors' code) and alpha is upsampled back.
    """
    def __init__(self,F_g,F_l,F_int):
        """
        F_g: Number of channels in gating signal (coarser decoder features, before upsampling)
        F_l: Number of channels in skip connection (from encoder ie downsampling region)
        F_int: Number of intermediate channels (usually F_l // 2)
        """
        super(AttentionGate,self).__init__()
        self.W_g=nn.Conv2d(F_g,F_int,kernel_size=1, stride=1,padding=0, bias=True)
        self.W_x=nn.Conv2d(F_l,F_int,kernel_size=2, stride=2,padding=0, bias=False)
        self.psi=nn.Sequential(
            nn.Conv2d(F_int,1,kernel_size=1, stride=1,padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu=nn.ReLU(inplace=True)

    def attention(self,x,g):
        alpha=self.psi(self.relu(self.W_x(x)+self.W_g(g)))
        return F.interpolate(alpha, size=x.shape[2:], mode='bilinear', align_corners=False)

    def forward(self,x,g):
        #x: original skip connection part, g:gating signal
        return x*self.attention(x,g)

class AttentionUNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, base_channels=64, block=DoubleConv):
        super(AttentionUNet, self).__init__()
        c = [base_channels * 2**i for i in range(5)]
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path.
        self.down_convolution_1 = block(in_channels, c[0])
        self.down_convolution_2 = block(c[0], c[1])
        self.down_convolution_3 = block(c[1], c[2])
        self.down_convolution_4 = block(c[2], c[3])
        self.down_convolution_5 = block(c[3], c[4])

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.up_convolution_1 = block(c[4], c[3])

        self.up_transpose_2 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.up_convolution_2 = block(c[3], c[2])

        self.up_transpose_3 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.up_convolution_3 = block(c[2], c[1])

        self.up_transpose_4 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)
        self.up_convolution_4 = block(c[1], c[0])

        self.out = nn.Conv2d(c[0], num_classes, kernel_size=1)

        # Attention Gates: gate i sees skip c[3-i] and the coarser decoder feature c[4-i] as g
        self.attn1 = AttentionGate(F_g=c[4], F_l=c[3], F_int=c[3] // 2)
        self.attn2 = AttentionGate(F_g=c[3], F_l=c[2], F_int=c[2] // 2)
        self.attn3 = AttentionGate(F_g=c[2], F_l=c[1], F_int=c[1] // 2)
        self.attn4 = AttentionGate(F_g=c[1], F_l=c[0], F_int=c[0] // 2)

    def forward(self, x):
        _check_input_size(x)

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

        # Decoder: each gate is driven by the coarser features before they are upsampled
        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([self.attn1(down_7, down_9), up_1], dim=1))

        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([self.attn2(down_5, x), up_2], dim=1))

        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([self.attn3(down_3, x), up_3], dim=1))

        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([self.attn4(down_1, x), up_4], dim=1))

        return self.out(x)



class FlexibleUNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, base_channels=64, block=DoubleConv):
        super(FlexibleUNet, self).__init__()
        c = [base_channels * 2**i for i in range(5)]
        self.max_pool2d = nn.MaxPool2d(2)

        self.down1 = block(in_channels, c[0])
        self.down2 = block(c[0], c[1])
        self.down3 = block(c[1], c[2])
        self.down4 = block(c[2], c[3])
        self.down5 = block(c[3], c[4])

        self.up1 = nn.ConvTranspose2d(c[4], c[3], 2, 2)
        self.conv1 = block(c[4], c[3])
        self.up2 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.conv2 = block(c[3], c[2])
        self.up3 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.conv3 = block(c[2], c[1])
        self.up4 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.conv4 = block(c[1], c[0])

        self.final = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        _check_input_size(x)

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
    """UNet++ (Zhou et al. 2018): X^{i,j} = H([X^{i,0}, ..., X^{i,j-1}, U(X^{i+1,j-1})]).

    With deep_supervision=True, forward returns a list of 4 logit maps from X^{0,1..4}
    (train on the mean of their losses; at inference average them, or keep an earlier
    head to prune the network).
    """
    def __init__(self, num_classes, in_channels=1, base_channels=64, block=DoubleConv, deep_supervision=False):
        super(FlexibleUNetPP, self).__init__()
        c = [base_channels * 2**i for i in range(5)]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        # Encoder blocks
        self.conv00 = block(in_channels, c[0])
        self.conv10 = block(c[0], c[1])
        self.conv20 = block(c[1], c[2])
        self.conv30 = block(c[2], c[3])
        self.conv40 = block(c[3], c[4])

        # Decoder blocks (Nested). X^{i,j} concatenates j same-level nodes plus one upsampled
        # node, and the upsampling halves channels to c[i], so its input has c[i] * (j + 1) channels.
        self.conv01 = block(c[0] * 2, c[0])
        self.conv11 = block(c[1] * 2, c[1])
        self.conv21 = block(c[2] * 2, c[2])
        self.conv31 = block(c[3] * 2, c[3])

        self.conv02 = block(c[0] * 3, c[0])
        self.conv12 = block(c[1] * 3, c[1])
        self.conv22 = block(c[2] * 3, c[2])

        self.conv03 = block(c[0] * 4, c[0])
        self.conv13 = block(c[1] * 4, c[1])

        self.conv04 = block(c[0] * 5, c[0])

        def up(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Upsample modules
        self.up01 = up(c[1], c[0])
        self.up11 = up(c[2], c[1])
        self.up21 = up(c[3], c[2])
        self.up31 = up(c[4], c[3])

        self.up02 = up(c[1], c[0])
        self.up12 = up(c[2], c[1])
        self.up22 = up(c[3], c[2])

        self.up03 = up(c[1], c[0])
        self.up13 = up(c[2], c[1])

        self.up04 = up(c[1], c[0])

        if deep_supervision:
            self.final = nn.ModuleList([nn.Conv2d(c[0], num_classes, kernel_size=1) for _ in range(4)])
        else:
            self.final = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        _check_input_size(x)

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

        if self.deep_supervision:
            return [head(node) for head, node in zip(self.final, [x01, x02, x03, x04])]
        return self.final(x04)
