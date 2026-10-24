"""Prepare a Kaggle (or any Linux GPU) machine to run this codebase. Run from the repo root:

    python kaggle/setup_kaggle.py

1. Installs the pinned albumentations (augmentation API must match the one fold 0 used); torch is kept as is.
2. Finds Kvasir-SEG (an attached Kaggle dataset, or ./Kvasir-SEG), or downloads the official zip.
3. Checks it is byte-identical to the copy fold 0 was trained on, and links it to ./Kvasir-SEG.
4. Prints the GPUs it sees.
"""
import hashlib
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
# md5 over (file name + bytes) of images/*.jpg then masks/*.jpg, sorted; computed on the local copy used for fold 0
EXPECTED_MD5 = "f64ec920fbcc0965d21f9bbe59af0de9"
# md5 of kvasir-seg.zip as served on 2026-10-22
EXPECTED_ZIP_MD5 = "6323d9094df93b35d43069a566ee1ca3"
SEARCH_DIRS = [ROOT, Path("/kaggle/input"), Path("/kaggle/working")]


def dataset_md5(root):
    h = hashlib.md5()
    for sub in ("images", "masks"):
        for p in sorted((root / sub).glob("*.jpg")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def find_kvasir():
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for images in base.rglob("images"):
            root = images.parent
            if (root / "masks").is_dir() and len(list(images.glob("*.jpg"))) == 1000:
                return root
    return None


def download_kvasir(dest=Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT):
    zip_path = dest / "kvasir-seg.zip"
    print(f"downloading {DATA_URL} (46 MB; needs Internet enabled in the notebook settings)")
    # datasets.simula.no serves an incomplete certificate chain (leaf from Sectigo "DV R36", wrong intermediates),
    # which Linux TLS stacks reject. Skip certificate verification and instead verify the zip's own md5
    # before extracting: that pins the exact bytes, which TLS alone would not.
    with urllib.request.urlopen(DATA_URL, context=ssl._create_unverified_context()) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if md5 != EXPECTED_ZIP_MD5:
        zip_path.unlink()
        sys.exit(f"downloaded zip md5 {md5} != {EXPECTED_ZIP_MD5}; refusing to extract")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()
    return find_kvasir()


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations==2.0.8"])

    data = find_kvasir() or download_kvasir()
    if data is None:
        sys.exit("Kvasir-SEG not found and download failed")
    md5 = dataset_md5(data)
    if md5 != EXPECTED_MD5:
        sys.exit(f"Kvasir-SEG at {data} differs from the fold-0 copy (md5 {md5} != {EXPECTED_MD5}); "
                 "results would not be comparable")
    link = ROOT / "Kvasir-SEG"
    if data.resolve() != link.resolve():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(data, target_is_directory=True)
    print(f"Kvasir-SEG OK at {data} (md5 matches fold 0); linked as {link}")

    import torch
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB")
    if not torch.cuda.is_available():
        print("WARNING: no GPU. In the notebook sidebar set Accelerator to 'GPU T4 x2'.")


if __name__ == "__main__":
    main()
