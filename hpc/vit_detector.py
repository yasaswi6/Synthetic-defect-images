"""The detector test on GPU: fine-tune the previous team's detector (ViT-base, in21k) on real TE
photos, optionally plus extra (fake) images, and score it on held-out photos.

  python hpc/vit_detector.py                         honest full-data baseline, folds 0-4
  python hpc/vit_detector.py --fewshot 10            AITK situation: <=10 real photos per defect
  python hpc/vit_detector.py --fewshot 10 --extra gen/txt2img/f{fold} --tag sdxl_txt2img
                                                     same, plus fakes ({fold} = the fold's own fakes;
                                                     a relative path is taken under $UROP_WORK)
  python hpc/vit_detector.py --dry_run               CPU smoke test with a tiny random model

Same recipe as the previous team's classification/train.py (ViT-base in21k, 224 px, flip,
+-10 deg rotation, colour jitter, AdamW, 20 epochs, keep the best epoch on validation), with
three fixes: validation is a separate fold (theirs was inside the test set), the learning
rate schedule steps every batch (theirs stepped once per epoch, so it stayed near 4e-5),
and the best epoch is picked on macro accuracy (each class counts equally) instead of plain
accuracy, which the 359 'Single' photos would dominate.

Appends one row per fold and seed to $UROP_RESULTS/vit_results.csv.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from splits_lib import CLASSES, DEFAULT_SPLITS, fewshot, fold_parts, load_splits  # noqa: E402

IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
MEAN = STD = [0.5, 0.5, 0.5]  # the normalisation this ViT was pre-trained with
TRAIN_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Photos(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = list(df.path), [IDX[c] for c in df.label], tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


def list_extra(root, cap):
    """Extra training images: root/<class>/*.png, class folders named as in CLASSES."""
    if not os.path.isdir(root):
        raise SystemExit(f"--extra folder not found: {root}")
    rows = []
    for c in CLASSES:
        d = os.path.join(root, c)
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
            rows += [(os.path.join(d, f), c) for f in files[:cap or None]]
    if not rows:
        raise SystemExit(f"no images in class folders under {root}")
    return pd.DataFrame(rows, columns=["path", "label"])


def build_model(dry):
    from transformers import ViTConfig, ViTForImageClassification
    if dry:
        cfg = ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                        image_size=224, patch_size=32, num_labels=len(CLASSES))
        return ViTForImageClassification(cfg)
    return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(CLASSES))


def scores(y, p):
    per = {c: float((p[y == i] == i).mean()) if (y == i).any() else float("nan") for i, c in enumerate(CLASSES)}
    ok = IDX["OK"]
    defect = y != ok
    escape = float((p[defect] == ok).mean()) if defect.any() else float("nan")       # defect called OK
    false_alarm = float((p[~defect] != ok).mean()) if (~defect).any() else float("nan")  # OK called defect
    return float(np.nanmean(list(per.values()))), per, escape, false_alarm


def run_fold(args, df, k, seed, device, amp_dtype):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train, val, test = fold_parts(df, k)
    train = fewshot(train, args.fewshot, args.ok_cap)[["path", "label"]]
    n_real = {c: int(n) for c, n in train.label.value_counts().items()}
    n_extra = 0
    if args.extra:
        root = args.extra.format(fold=k)
        root = root if os.path.isabs(root) else os.path.join(os.environ.get("UROP_WORK", ""), root)
        extra = list_extra(root, args.extra_per_class)
        n_extra = len(extra)
        train = pd.concat([train, extra], ignore_index=True)
    if args.dry_run:
        train, val, test = (d.groupby("label").head(3) for d in (train, val, test))

    workers = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 2)
    pin = device.type == "cuda"
    tr = DataLoader(Photos(train, TRAIN_TF), batch_size=args.bs, shuffle=True, num_workers=workers, pin_memory=pin)
    va = DataLoader(Photos(val, EVAL_TF), batch_size=64, num_workers=workers, pin_memory=pin)
    te = DataLoader(Photos(test, EVAL_TF), batch_size=64, num_workers=workers, pin_memory=pin)

    model = build_model(args.dry_run).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1)
    weight = None
    if args.balance:  # optional: each class weighs the same in the loss
        cnt = train.label.value_counts()
        weight = torch.tensor([len(train) / (len(CLASSES) * cnt.get(c, 1)) for c in CLASSES]).float().to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=weight)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        ys, ps = [], []
        for x, y in loader:
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                ps.append(model(pixel_values=x.to(device)).logits.argmax(1).cpu())
            ys.append(y)
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    best = (-1.0, None, 0)
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                loss = lossf(model(pixel_values=x).logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        vm = scores(*predict(va))[0]
        if vm > best[0]:
            best = (vm, {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}, ep + 1)
        print(f"  fold {k} seed {seed} epoch {ep + 1:2d}/{args.epochs}  loss {total / max(len(tr), 1):.3f}  "
              f"val macro acc {vm:.3f}", flush=True)

    model.load_state_dict(best[1])
    macro, per, esc, fa = scores(*predict(te))
    return dict(tag=args.tag, fold=k, seed=seed, fewshot=args.fewshot, extra=args.extra, n_extra=n_extra,
                n_real_train=json.dumps(n_real), n_test=len(test), best_epoch=best[2], val_macro=round(best[0], 4),
                macro_acc=round(macro, 4), escape_rate=round(esc, 4), false_alarm=round(fa, 4),
                **{f"acc_{c}": round(v, 4) for c, v in per.items()},
                time=time.strftime("%Y-%m-%d %H:%M"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default=DEFAULT_SPLITS)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fewshot", type=int, default=0, help="max real photos per defect class in training (0 = all)")
    ap.add_argument("--ok_cap", type=int, default=100, help="max OK photos in few-shot mode")
    ap.add_argument("--extra", default="", help="folder of extra training images, <class>/ subfolders; {fold} allowed")
    ap.add_argument("--extra_per_class", type=int, default=0, help="use at most this many extra images per class")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--balance", action="store_true", help="class-weighted loss")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(os.environ.get("UROP_RESULTS", os.path.join(HERE, "results")),
                                                  "vit_results.csv"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.folds, args.bs = 1, args.folds[:1], 4
    args.tag = args.tag or ("fewshot%d" % args.fewshot if args.fewshot else "full") + ("+extra" if args.extra else "")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("GPU:", torch.cuda.get_device_name(0), "| autocast", amp_dtype)
    df = load_splits(args.splits)
    print(f"{len(df)} photos, tag '{args.tag}', folds {args.folds}, seeds {args.seeds}", flush=True)

    rows = []
    for seed in args.seeds:
        for k in args.folds:
            row = run_fold(args, df, k, seed, device, amp_dtype)
            rows.append(row)
            print(f"fold {k} seed {seed}: test macro acc {row['macro_acc']:.3f}  escape {row['escape_rate']:.3f}  "
                  f"false alarm {row['false_alarm']:.3f}", flush=True)
            if not args.dry_run:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
    res = pd.DataFrame(rows)
    print(f"\n{args.tag}: macro accuracy {res.macro_acc.mean():.3f} +- {res.macro_acc.std(ddof=0):.3f} "
          f"over {len(res)} runs" + ("" if args.dry_run else f"  (saved to {args.out})"))


if __name__ == "__main__":
    main()
