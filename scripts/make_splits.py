"""Make leakage-aware 5-fold splits for Kvasir-SEG and write them to splits/kvasir_5fold.json.

Near-duplicate images (same polyp from the same video) are grouped so a group never spans two folds:
pairs with ResNet50 (ImageNet) cosine similarity >= --cos are linked, then linked components form groups.
The 0.89 default was chosen by viewing sample pairs per similarity band (runs/audit/embedding_pairs.png):
>= 0.90 were almost all the same polyp, 0.88-0.89 almost all different. Folds are stratified by polyp size.

Usage: python scripts/make_splits.py [--root Kvasir-SEG] [--cos 0.89] [--seed 0]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold


def embed(paths, batch_size=50):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    net = torchvision.models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    net.eval().to(device)
    tf = weights.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
            feats.append(torch.nn.functional.normalize(net(x), dim=1).cpu())
    return torch.cat(feats).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Kvasir-SEG")
    ap.add_argument("--cos", type=float, default=0.89)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits/kvasir_5fold.json")
    args = ap.parse_args()
    root = Path(args.root)

    names = sorted(p.name for p in (root / "images").glob("*.jpg"))
    feats = embed([root / "images" / n for n in names])

    sim = feats @ feats.T
    n_groups, group = connected_components(csr_matrix(sim >= args.cos), directed=False)

    area = np.array([(np.asarray(Image.open(root / "masks" / n))[..., 0] >= 128).mean() for n in names])
    size_bin = np.digitize(area, np.percentile(area, [100 / 3, 200 / 3]))  # small / medium / large polyps

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold = np.empty(len(names), dtype=int)
    for k, (_, test_idx) in enumerate(cv.split(names, size_bin, group)):
        fold[test_idx] = k

    sizes = Counter(Counter(group).values())
    print(f"{len(names)} images -> {n_groups} groups; group sizes: {sorted(sizes.items())}")
    for k in range(args.folds):
        m = fold == k
        print(f"fold {k}: {m.sum()} images, {len(set(group[m]))} groups, "
              f"median polyp area {np.median(area[m]):.1%}, size bins {np.bincount(size_bin[m], minlength=3).tolist()}")
    assert all(len(set(fold[group == g])) == 1 for g in range(n_groups)), "a group spans two folds"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "Kvasir-SEG",
        "method": f"ResNet50-IMAGENET1K_V2 cosine >= {args.cos} -> connected components; "
                  f"StratifiedGroupKFold(n_splits={args.folds}, shuffle=True, random_state={args.seed}) on polyp-area terciles",
        "folds": {n: int(f) for n, f in zip(names, fold)},
        "groups": {n: int(g) for n, g in zip(names, group)},
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
