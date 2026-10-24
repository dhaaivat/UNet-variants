import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def load_split(path, test_fold, n_folds=5):
    """Fold `test_fold` is the test set, the next fold is validation, the remaining folds train."""
    folds = json.loads(Path(path).read_text())["folds"]
    val_fold = (test_fold + 1) % n_folds
    split = {"train": [], "val": [], "test": []}
    for name, f in sorted(folds.items()):
        key = "test" if f == test_fold else "val" if f == val_fold else "train"
        split[key].append(name)
    return split


def train_transform(size):
    # Endoscopy has no canonical orientation, so flips and 90-degree rotations are label-preserving
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
        A.Normalize(),  # ImageNet mean/std; the inputs are RGB photographs
        ToTensorV2()
    ])


def eval_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2()
    ])


def read_pair(root, name):
    root = Path(root)
    image = np.asarray(Image.open(root / "images" / name).convert("RGB"))
    mask = (np.asarray(Image.open(root / "masks" / name).convert("L")) >= 128).astype(np.uint8)
    return image, mask


def resize_pair(image, mask, size):
    # Same interpolation as A.Resize defaults, so a later A.Resize(size, size) is a no-op
    return (cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR),
            cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))


class KvasirDataset(Dataset):
    """Returns (image [3,S,S] float, mask [1,S,S] float in {0,1}).

    With return_original=True also returns the full-resolution mask and the file name,
    so test metrics can be computed at the original image size (use batch_size=1).
    With preload_size=S, every pair is decoded and resized to SxS once, up front, which
    removes JPEG decoding from the per-epoch cost without changing any output.
    """
    def __init__(self, root, names, transform, return_original=False, preload_size=None):
        self.root = root
        self.names = list(names)
        self.transform = transform
        self.return_original = return_original
        self.cache = None
        if preload_size is not None and not return_original:
            self.cache = [resize_pair(*read_pair(root, n), preload_size) for n in self.names]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        image, mask = self.cache[i] if self.cache is not None else read_pair(self.root, self.names[i])
        out = self.transform(image=image, mask=mask)
        x, y = out["image"], out["mask"].unsqueeze(0).float()
        if self.return_original:
            return x, y, torch.from_numpy(mask), self.names[i]
        return x, y
