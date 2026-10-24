from pathlib import Path

import numpy as np
import pytest
import torch

from utils.data import KvasirDataset, eval_transform, load_split, train_transform
from utils.losses import DiceBCELoss, dice_iou

ROOT, SPLITS = Path("Kvasir-SEG"), Path("splits/kvasir_5fold.json")
needs_data = pytest.mark.skipif(not (ROOT.exists() and SPLITS.exists()), reason="Kvasir-SEG not present")


def test_dice_iou_known_values():
    pred = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
    target = torch.zeros_like(pred)
    pred[0, 0, :2], target[0, 0, :1] = True, True  # 8 predicted, 4 true, 4 overlap
    target[1, 0, 0, 0] = True                      # missed entirely
    dice, iou = dice_iou(pred, target)              # image 2: both empty
    torch.testing.assert_close(dice, torch.tensor([8 / 12, 0.0, 1.0]))
    torch.testing.assert_close(iou, torch.tensor([4 / 8, 0.0, 1.0]))


def test_loss_low_for_perfect_and_high_for_inverted_prediction():
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()
    loss = DiceBCELoss()
    perfect, inverted = loss(target * 40 - 20, target), loss(20 - target * 40, target)
    assert perfect < 0.01 and inverted > 10
    torch.testing.assert_close(loss([target * 40 - 20] * 4, target), perfect)  # deep-supervision list


@needs_data
@pytest.mark.parametrize("fold", range(5))
def test_split_is_disjoint_and_groups_do_not_leak(fold):
    import json
    split = load_split(SPLITS, fold)
    sets = {k: set(v) for k, v in split.items()}
    assert sum(map(len, sets.values())) == 1000
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    groups = json.loads(SPLITS.read_text())["groups"]
    g = {k: {groups[n] for n in v} for k, v in sets.items()}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"])


@needs_data
def test_dataset_shapes_and_binary_masks():
    names = load_split(SPLITS, 0)["train"][:8]
    for tf in (train_transform(352), eval_transform(352)):
        for x, y in KvasirDataset(ROOT, names, tf):
            assert x.shape == (3, 352, 352) and x.dtype == torch.float32
            assert y.shape == (1, 352, 352) and set(y.unique().tolist()) <= {0.0, 1.0}
    x, y, orig, name = KvasirDataset(ROOT, names[:1], eval_transform(352), return_original=True)[0]
    assert orig.ndim == 2 and orig.max() == 1 and name == names[0]


@needs_data
def test_augmentation_moves_image_and_mask_together():
    """Burn the mask into the image as a pure-green channel, augment, and check they still line up."""
    from utils.data import read_pair
    import albumentations as A
    name = load_split(SPLITS, 0)["train"][0]
    image, mask = read_pair(ROOT, name)
    marked = np.zeros_like(image)
    marked[..., 1] = mask * 255
    geometric = A.Compose([t for t in train_transform(352).transforms
                           if not isinstance(t, (A.ColorJitter, A.Normalize)) and type(t).__name__ != "ToTensorV2"])
    for seed in range(10):
        geometric.set_random_seed(seed)
        out = geometric(image=marked, mask=mask)
        agree = ((out["image"][..., 1] > 127) == (out["mask"] > 0)).mean()
        assert agree > 0.99, f"seed {seed}: image and mask disagree on {1 - agree:.2%} of pixels"


@needs_data
def test_preload_gives_identical_samples():
    """Resizing once up front must not change what the model sees (A.Resize becomes a no-op)."""
    names = load_split(SPLITS, 0)["val"][:6]
    lazy = KvasirDataset(ROOT, names, eval_transform(256))
    cached = KvasirDataset(ROOT, names, eval_transform(256), preload_size=256)
    for (x1, y1), (x2, y2) in zip(lazy, cached):
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
    tf = train_transform(256)
    for i in range(len(names)):
        tf.set_random_seed(i)
        x1, y1 = lazy.__class__(ROOT, names, tf)[i]
        tf.set_random_seed(i)
        x2, y2 = KvasirDataset(ROOT, names, tf, preload_size=256)[i]
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
