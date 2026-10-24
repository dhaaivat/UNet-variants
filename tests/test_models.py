import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UNet_models import (
    AttentionGate, AttentionUNet, DoubleConv, FlexibleUNet, FlexibleUNetPP, ResidualBlock,
)


class PaperConv(nn.Module):
    """(3x3 conv with bias -> ReLU) x 2, no BatchNorm: the block used for the UNet++ paper's counts."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


MODELS = [
    pytest.param(lambda **kw: FlexibleUNet(**kw), id="unet"),
    pytest.param(lambda **kw: AttentionUNet(**kw), id="attention_unet"),
    pytest.param(lambda **kw: FlexibleUNetPP(**kw), id="unetpp"),
]
BLOCKS = [DoubleConv, ResidualBlock]


@pytest.mark.parametrize("make", MODELS)
@pytest.mark.parametrize("block", BLOCKS, ids=["double_conv", "residual"])
@pytest.mark.parametrize("in_channels,num_classes", [(1, 1), (1, 3), (3, 2)])
def test_output_shape(make, block, in_channels, num_classes):
    model = make(num_classes=num_classes, in_channels=in_channels, base_channels=8, block=block).eval()
    x = torch.randn(2, in_channels, 64, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, num_classes, 64, 96)


@pytest.mark.parametrize("make", MODELS)
def test_rejects_size_not_divisible_by_16(make):
    model = make(num_classes=1, base_channels=8).eval()
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.randn(1, 1, 72, 64))


def test_unetpp_deep_supervision_returns_four_full_res_heads():
    model = FlexibleUNetPP(num_classes=2, base_channels=8, deep_supervision=True).eval()
    with torch.no_grad():
        outs = model(torch.randn(1, 1, 64, 64))
    assert isinstance(outs, list) and len(outs) == 4
    assert all(o.shape == (1, 2, 64, 64) for o in outs)


# Zhou et al. 2018 (UNet++, DLMIA), Table 2: filters 32..512, 1 input / 1 output channel.
def test_unet_param_count_matches_unetpp_paper():
    model = FlexibleUNet(num_classes=1, in_channels=1, base_channels=32, block=PaperConv)
    assert round(n_params(model) / 1e6, 2) == 7.76


def test_unetpp_param_count_matches_unetpp_paper():
    for ds in (False, True):
        model = FlexibleUNetPP(num_classes=1, in_channels=1, base_channels=32, block=PaperConv, deep_supervision=ds)
        assert round(n_params(model) / 1e6, 2) == 9.04


def test_attention_gate_equations():
    torch.manual_seed(0)
    gate = AttentionGate(F_g=16, F_l=8, F_int=4)
    x = torch.randn(2, 8, 32, 32)
    g = torch.randn(2, 16, 16, 16, requires_grad=True)

    alpha = gate.attention(x, g)
    assert alpha.shape == (2, 1, 32, 32)
    assert alpha.min() > 0 and alpha.max() < 1

    # Oktay et al. eqs. 1-2, written out by hand
    q = F.relu(F.conv2d(x, gate.W_x.weight, stride=2) + F.conv2d(g, gate.W_g.weight, gate.W_g.bias))
    expected = torch.sigmoid(F.conv2d(q, gate.psi[0].weight, gate.psi[0].bias))
    expected = F.interpolate(expected, size=(32, 32), mode="bilinear", align_corners=False)
    torch.testing.assert_close(alpha, expected)

    out = gate(x, g)
    torch.testing.assert_close(out, x * expected)
    out.sum().backward()
    assert g.grad is not None and g.grad.abs().sum() > 0


def test_attention_unet_gates_use_coarser_gating_signal():
    """Every gate must receive g at half the resolution of its skip connection."""
    model = AttentionUNet(num_classes=1, base_channels=8).eval()
    seen = []
    for gate in (model.attn1, model.attn2, model.attn3, model.attn4):
        gate.register_forward_hook(lambda m, inp, out: seen.append((inp[0].shape[-1], inp[1].shape[-1])))
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))
    assert seen == [(8, 4), (16, 8), (32, 16), (64, 32)]


@pytest.mark.parametrize("make", MODELS)
def test_overfits_single_batch(make):
    """A correct network must be able to memorise one batch; failure means a wiring or gradient bug."""
    torch.manual_seed(0)
    model = make(num_classes=1, in_channels=1, base_channels=8)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    mask = (((yy - 24) ** 2 + (xx - 40) ** 2) < 12 ** 2).float()[None, None].repeat(2, 1, 1, 1)
    image = mask + 0.3 * torch.randn_like(mask)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(image), mask)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = (model(image) > 0).float()
    dice = (2 * (pred * mask).sum() / (pred.sum() + mask.sum())).item()
    assert dice > 0.95, f"dice={dice:.3f}"
