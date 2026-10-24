"""Train every model on the given folds, spreading folds across GPUs (one process per GPU).

Each GPU works through its folds one after another, and all three models for a fold one after another.
Runs whose summary.json already exists are skipped, so an interrupted job can simply be relaunched.

Examples:
  python scripts/run_folds.py --folds 1 2 3 4                 # one device (Mac / single GPU), sequential
  python scripts/run_folds.py --folds 1 2 3 4 --gpus 0 1      # Kaggle T4 x2: GPU 0 gets folds 1,3; GPU 1 gets 2,4
  python scripts/run_folds.py --folds 1 2 3 4 --gpus 0 1 -- --amp --workers 2   # extra args go to train.py
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["unet", "attention_unet", "unetpp"]


def run_queue(folds, gpu, out, extra):
    log_dir = Path(out) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for fold in folds:
        for model in MODELS:
            if (Path(out) / f"{model}_double_conv" / f"fold{fold}" / "summary.json").exists():
                print(f"[gpu {gpu}] skip {model} fold {fold}: already done", flush=True)
                continue
            print(f"[gpu {gpu}] {time.strftime('%F %T')} start {model} fold {fold}", flush=True)
            with open(log_dir / f"{model}_fold{fold}.log", "w") as log, open(log_dir / f"{model}_fold{fold}.err", "w") as err:
                code = subprocess.call([sys.executable, "train.py", "--model", model, "--fold", str(fold), "--out", out, *extra],
                                       cwd=ROOT, env=env, stdout=log, stderr=err)
            print(f"[gpu {gpu}] {time.strftime('%F %T')} end {model} fold {fold} exit {code}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="CUDA device ids; omit to use the default device")
    ap.add_argument("--out", default="runs/kvasir")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --, arguments passed through to train.py")
    args = ap.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    if not args.gpus:
        run_queue(args.folds, None, args.out, extra)
        return
    # Round-robin folds over GPUs; one thread per GPU, each launching its own train.py processes
    threads = [threading.Thread(target=run_queue, args=(args.folds[k::len(args.gpus)], gpu, args.out, extra))
               for k, gpu in enumerate(args.gpus) if args.folds[k::len(args.gpus)]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
