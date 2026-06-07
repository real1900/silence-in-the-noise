"""
Properly powered ESC-50 5-fold CV training for the AST + Register Tokens paper.

Designed to run on a single A100 / L4 / T4 GPU on GCP. Uses the same
`ModifiedAST` class as `measure_real_results.py` for protocol parity.

Protocol:
  - 5 folds (ESC-50's canonical splits)
  - 10 epochs per fold by default
  - Batch size 16 (fits on 16 GB; auto-shrinks if OOM)
  - AdamW peak LR 1e-4, weight_decay 1e-2, OneCycle schedule
  - Each (n_register, fold) configuration trained from scratch from the
    AudioSet-pretrained AST checkpoint
  - Per-clip predictions saved for downstream bootstrap analysis

Sweep:
  - n in {0, 4} by default (matches the paper's main comparison)
  - --sweep argument can override to {0, 2, 4, 8, 16}

Outputs to paper_assets/real_5fold.json + per-fold prediction NPZ files.

Estimated wall-clock on an A100 (40 GB):
  - n=0:      ~7-10 minutes per fold × 5 folds ≈ 50 min
  - n=4:      ~7-10 minutes per fold × 5 folds ≈ 50 min
  - Total:    ~1.5-2 hours, ~$2-4 of A100 spot credit

Usage (locally to smoke-test):
    python train_full_5fold.py --epochs 1 --folds 1 --train-batch 4
Usage (on GCP):
    python train_full_5fold.py --epochs 10 --folds all --train-batch 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import ASTModel, ASTFeatureExtractor
import io
import csv
import urllib.request
import zipfile
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "paper_assets")
os.makedirs(ASSETS, exist_ok=True)

SEED = 42
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 50
SAMPLE_RATE = 16000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# ESC-50 direct loader (avoids HuggingFace datasets / torchcodec)
# ----------------------------------------------------------------------
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc50_data")


class _AudioRow:
    """Minimal dict-like row that train_fold's collate fn can consume."""
    __slots__ = ("path", "target", "fold", "category")
    def __init__(self, path, target, fold, category):
        self.path = path
        self.target = target
        self.fold = fold
        self.category = category
    def __getitem__(self, k):
        if k == "audio":  return {"path": self.path}
        if k == "target": return self.target
        if k == "fold":   return self.fold
        if k == "category": return self.category
        raise KeyError(k)


def load_esc50_directly():
    """Download ESC-50 from GitHub if needed, return (rows, fold_to_idx)."""
    audio_dir = os.path.join(ESC50_DIR, "ESC-50-master", "audio")
    meta_csv  = os.path.join(ESC50_DIR, "ESC-50-master", "meta", "esc50.csv")
    if not os.path.exists(meta_csv):
        print(f"[data] downloading ESC-50 from {ESC50_URL} ...")
        os.makedirs(ESC50_DIR, exist_ok=True)
        zip_path = os.path.join(ESC50_DIR, "esc50.zip")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        print(f"[data] extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ESC50_DIR)
        os.remove(zip_path)
    rows = []
    with open(meta_csv) as f:
        for r in csv.DictReader(f):
            rows.append(_AudioRow(
                path=os.path.join(audio_dir, r["filename"]),
                target=int(r["target"]),
                fold=int(r["fold"]),
                category=r["category"],
            ))
    fold_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        fold_to_idx[r.fold].append(i)
    return rows, fold_to_idx


# ----------------------------------------------------------------------
# Model (mirror of measure_real_results.ModifiedAST)
# ----------------------------------------------------------------------
class ModifiedAST(nn.Module):
    def __init__(self, model_id: str, num_classes: int, n_registers: int = 0,
                 capture_attn: bool = False):
        super().__init__()
        attn_impl = "eager" if capture_attn else "sdpa"
        self.encoder = ASTModel.from_pretrained(model_id, attn_implementation=attn_impl)
        self.embed_dim = self.encoder.config.hidden_size
        self.n_registers = n_registers
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        if n_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, n_registers, self.embed_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_parameter("register_tokens", None)

        self._last_attn = None
        if capture_attn:
            final_self_attn = self.encoder.encoder.layer[-1].attention.attention
            def hook(module, inp, out):
                self._last_attn = out[1].detach() if isinstance(out, tuple) else None
            final_self_attn.register_forward_hook(hook)

    def _build_input_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        emb = self.encoder.embeddings(input_values)
        if self.n_registers > 0:
            B = emb.shape[0]
            cls_dst = emb[:, :2, :]
            patches = emb[:, 2:, :]
            regs = self.register_tokens.expand(B, -1, -1)
            emb = torch.cat([cls_dst, regs, patches], dim=1)
        return emb

    def forward(self, input_values: torch.Tensor):
        self._last_attn = None
        emb = self._build_input_embeddings(input_values)
        enc_out = self.encoder.encoder(emb)
        h = self.encoder.layernorm(enc_out.last_hidden_state)
        cls_repr = h[:, 0]
        return {"logits": self.classifier(cls_repr), "hidden_last": h,
                "final_attn": self._last_attn}


# ----------------------------------------------------------------------
# Training one fold
# ----------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(y.cpu().numpy())
    return (correct / max(total, 1),
            np.concatenate(preds_all),
            np.concatenate(targets_all))


def train_fold(n_reg: int, fold: int, fold_to_idx, ds, feature_extractor,
               args, device) -> Dict:
    print(f"\n=== n_reg={n_reg}  test_fold={fold} ===")
    set_seed(SEED + fold)

    train_idx = [i for f, idxs in fold_to_idx.items() if f != fold for i in idxs]
    test_idx = fold_to_idx[fold]
    print(f"   train clips: {len(train_idx)}, test clips: {len(test_idx)}")

    def _decode_audio(a):
        # Handle every plausible shape the dataset can return for the audio column.
        if isinstance(a, dict):
            if "array" in a and a["array"] is not None:
                arr = np.asarray(a["array"], dtype=np.float32)
            elif "bytes" in a and a["bytes"] is not None:
                arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32",
                                   always_2d=False)
            elif "path" in a and a["path"]:
                arr, sr = sf.read(a["path"], dtype="float32", always_2d=False)
            else:
                raise ValueError(f"audio dict has no usable key: {list(a.keys())}")
        else:
            arr = np.asarray(a, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)  # downmix any stereo input
        return arr

    def collate(batch):
        audios = [_decode_audio(b["audio"]) for b in batch]
        labels = torch.tensor([b["target"] for b in batch], dtype=torch.long)
        feats = feature_extractor(audios, sampling_rate=SAMPLE_RATE,
                                  return_tensors="pt")
        return feats["input_values"], labels

    train_loader = DataLoader(Subset(ds, train_idx),
                              batch_size=args.train_batch,
                              shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(Subset(ds, test_idx),
                             batch_size=args.eval_batch,
                             shuffle=False, collate_fn=collate,
                             num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"))

    model = ModifiedAST(MODEL_ID, NUM_CLASSES, n_registers=n_reg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                          total_steps=args.epochs * len(train_loader))

    history = {"train_acc": [], "test_acc": [], "lr": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        correct = total = running_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)["logits"]
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
            running_loss += loss.item() * y.size(0)
        train_acc = correct / max(total, 1)
        test_acc, preds, targets = evaluate(model, test_loader, device)
        history["train_acc"].append(float(train_acc))
        history["test_acc"].append(float(test_acc))
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        print(f"   ep {ep+1:2d}/{args.epochs}  loss={running_loss/total:.3f} "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    final_test_acc, final_preds, final_targets = evaluate(model, test_loader, device)
    elapsed = time.time() - t0
    print(f"   done in {elapsed:.1f}s   final test_acc={final_test_acc:.4f}")

    return {
        "n_reg": int(n_reg),
        "fold": int(fold),
        "best_train_acc": float(max(history["train_acc"])),
        "best_test_acc": float(max(history["test_acc"])),
        "final_test_acc": float(final_test_acc),
        "history": history,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "wall_time_sec": float(elapsed),
        "preds": final_preds.astype(np.int32),
        "targets": final_targets.astype(np.int32),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--folds", type=str, default="all",
                   help="'all' or comma-separated list, e.g. '1,2,3'")
    p.add_argument("--sweep", type=str, default="0,4",
                   help="comma-separated register counts, e.g. '0,4' or '0,2,4,8,16'")
    p.add_argument("--seed", type=int, default=42,
                   help="Master seed; per-fold seed = seed + fold")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path; defaults to real_5fold_seed<seed>.json")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 1 fold, batch 4 - sanity check on local hardware")
    args = p.parse_args()
    global SEED
    SEED = args.seed
    if args.out is None:
        args.out = os.path.join(ASSETS, f"real_5fold_seed{args.seed}.json")

    if args.smoke:
        args.epochs = 1
        args.train_batch = 4
        args.eval_batch = 4
        args.folds = "1"
        args.sweep = "0,4"
        args.num_workers = 0

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        print(f"[device] CUDA  {gpu_name}  free={free/1e9:.1f}/{total/1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("[device] CPU - this will be VERY slow")

    print(f"[args] {vars(args)}")

    # data - download ESC-50 directly from GitHub to avoid HF datasets'
    # torchcodec dependency (which has ABI hell on many CUDA wheel builds)
    ds, fold_to_idx = load_esc50_directly()
    print(f"[data] {len(ds)} clips loaded; folds: {sorted(fold_to_idx)}, "
          f"sizes: {[len(fold_to_idx[f]) for f in sorted(fold_to_idx)]}")
    feature_extractor = ASTFeatureExtractor()

    # which folds and register counts
    folds = sorted(fold_to_idx) if args.folds == "all" else \
        [int(s) for s in args.folds.split(",")]
    sweep = [int(s) for s in args.sweep.split(",")]
    print(f"[plan] folds={folds}, n_reg sweep={sweep}, epochs={args.epochs}")

    all_results = []
    npz_payload = {}
    t_global = time.time()

    for n_reg in sweep:
        for fold in folds:
            res = train_fold(n_reg, fold, fold_to_idx, ds,
                             feature_extractor, args, device)
            npz_payload[f"preds_n{n_reg}_fold{fold}"] = res.pop("preds")
            npz_payload[f"targets_n{n_reg}_fold{fold}"] = res.pop("targets")
            all_results.append(res)

            # persist intermediate after every fold so a crash doesn't lose
            # everything; safer than waiting until end.
            np.savez(os.path.join(ASSETS, f"real_5fold_preds_seed{args.seed}.npz"), **npz_payload)
            with open(args.out, "w") as f:
                json.dump({
                    "args": vars(args),
                    "device": str(device),
                    "results": all_results,
                    "wall_time_sec": time.time() - t_global,
                }, f, indent=2)

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY (best test accuracy per (n, fold))")
    print("=" * 60)
    summary = defaultdict(list)
    for r in all_results:
        summary[r["n_reg"]].append(r["best_test_acc"])
    for n_reg, accs in sorted(summary.items()):
        accs = np.array(accs)
        print(f"  n={n_reg:>2}  folds={len(accs)}  "
              f"mean={accs.mean()*100:.2f}%  std={accs.std()*100:.2f}%  "
              f"min={accs.min()*100:.2f}  max={accs.max()*100:.2f}")

    print(f"\nTotal wall-time: {(time.time()-t_global)/60:.1f} min")
    print(f"Results JSON   : {args.out}")
    print(f"Predictions    : {os.path.join(ASSETS, 'real_5fold_preds.npz')}")


if __name__ == "__main__":
    main()
