import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE + soft Dice for binary segmentation with a single logit channel.

    Accepts a list of logit maps (UNet++ deep supervision) and averages the loss over them.
    Soft Dice is computed per image, then averaged, so small polyps count as much as large ones.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            return sum(self(l, target) for l in logits) / len(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + dice.mean()


def dice_iou(pred, target):
    """Per-image Dice and IoU for boolean tensors [N, ...]. Both empty counts as a perfect match."""
    pred, target = pred.flatten(1).bool(), target.flatten(1).bool()
    inter = (pred & target).sum(1).float()
    p, t = pred.sum(1).float(), target.sum(1).float()
    union = p + t - inter
    empty = (p + t) == 0
    dice = torch.where(empty, torch.ones_like(inter), 2 * inter / (p + t).clamp(min=1))
    iou = torch.where(empty, torch.ones_like(inter), inter / union.clamp(min=1))
    return dice, iou
