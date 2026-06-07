"""
Real-data measurement script for the AST + Register Token paper.

In transformers >= 5.x the AST encoder no longer plumbs `output_attentions`
through to the caller; instead the attention probabilities are produced by
each ASTSelfAttention module and immediately discarded. To capture them we
register a forward hook on the final layer's self-attention module.

Outputs to paper_assets/real_results.json so the LaTeX paper and notebook
can be updated with verified, real numbers from the actual pretrained AST
checkpoint and ESC-50 dataset.
"""
from __future__ import annotations

import json
import os
import random
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset, Audio
from transformers import ASTModel, ASTFeatureExtractor

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "paper_assets")
os.makedirs(ASSETS, exist_ok=True)
OUT_JSON = os.path.join(ASSETS, "real_results.json")

SEED = 42
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 50
SAMPLE_RATE = 16000
ARTIFACT_THRESHOLD_IQR = 2.5

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[measure_real_results] device={DEVICE}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Modified AST architecture with attention capture via forward hook.
# ----------------------------------------------------------------------
class ModifiedAST(nn.Module):
    def __init__(self, model_id: str, num_classes: int, n_registers: int = 4):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(model_id, attn_implementation="eager")
        self.embed_dim = self.encoder.config.hidden_size
        self.n_registers = n_registers
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        if n_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_registers, self.embed_dim))
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_parameter("register_tokens", None)

        # Forward hook on the final layer's self-attention module to capture
        # the attention probability tensor.
        self._last_attn = None
        final_self_attn = self.encoder.encoder.layer[-1].attention.attention

        def hook(module, inp, out):
            self._last_attn = out[1].detach() if isinstance(out, tuple) else None

        self._hook_handle = final_self_attn.register_forward_hook(hook)

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
        logits = self.classifier(cls_repr)
        return {
            "logits": logits,
            "hidden_last": h,
            "final_attn": self._last_attn,  # (B, H, S, S) or None
        }

    def patch_index_offset(self) -> int:
        return 2 + self.n_registers


# ----------------------------------------------------------------------
# Diagnostic primitives
# ----------------------------------------------------------------------
@torch.no_grad()
def token_norm_stats(model: ModifiedAST, x: torch.Tensor) -> Dict[str, np.ndarray]:
    out = model(x)
    h = out["hidden_last"]
    norms = h.norm(dim=-1).cpu().numpy()
    n_reg = model.n_registers
    return {
        "special": norms[:, :2].reshape(-1),
        "register": norms[:, 2:2 + n_reg].reshape(-1) if n_reg > 0 else np.array([]),
        "patch": norms[:, 2 + n_reg:].reshape(-1),
        "all": norms.reshape(-1),
    }


@torch.no_grad()
def register_attention_mass(model: ModifiedAST, x: torch.Tensor) -> float:
    if model.n_registers == 0:
        return 0.0
    out = model(x)
    A = out["final_attn"]
    if A is None:
        return float("nan")
    reg_slice = slice(2, 2 + model.n_registers)
    mass = A[..., reg_slice].sum(dim=-1)
    return float(mass.mean().item())


@torch.no_grad()
def patch_attention_frobenius(model: ModifiedAST, x: torch.Tensor) -> float:
    out = model(x)
    A = out["final_attn"]
    if A is None:
        return float("nan")
    A_h = A.mean(dim=1)  # (B, S, S) head-averaged
    p = model.patch_index_offset()
    A_pp = A_h[:, p:, p:]
    fro = A_pp.flatten(start_dim=1).norm(dim=-1)
    return float(fro.mean().item())


@torch.no_grad()
def collect_attention_map(model: ModifiedAST, x: torch.Tensor) -> np.ndarray:
    out = model(x)
    A = out["final_attn"]
    if A is None:
        raise RuntimeError("attention not captured")
    return A.mean(dim=1)[0].cpu().numpy()


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def main() -> None:
    set_seed()
    print("[measure_real_results] loading ESC-50 ...")
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    fold_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, fold in enumerate(ds["fold"]):
        fold_to_idx[int(fold)].append(i)
    id2label = {int(t): n for t, n in zip(ds["target"], ds["category"])}
    label_names = [id2label[i] for i in range(NUM_CLASSES)]

    feature_extractor = ASTFeatureExtractor()

    def collate(batch):
        audios = [b["audio"]["array"] for b in batch]
        labels = torch.tensor([b["target"] for b in batch], dtype=torch.long)
        feats = feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return feats["input_values"], labels

    # ------------------------------------------------------------------
    # 1. Diagnostic measurements on PRETRAINED models.
    # ------------------------------------------------------------------
    print("[measure_real_results] diagnostic measurements on pretrained models ...")
    set_seed()
    diag_loader = DataLoader(
        Subset(ds, list(range(64))), batch_size=4, collate_fn=collate
    )

    diag_results: Dict[str, Dict[str, float]] = {}
    norm_dump: Dict[str, np.ndarray] = {}

    for n_reg in [0, 2, 4, 8, 16]:
        print(f"  - building Modified AST with n={n_reg}")
        set_seed()
        model = ModifiedAST(MODEL_ID, NUM_CLASSES, n_registers=n_reg).to(DEVICE).eval()
        all_norms = {"special": [], "register": [], "patch": []}
        fro_vals, reg_mass_vals = [], []
        for x, _ in diag_loader:
            x = x.to(DEVICE)
            n_stats = token_norm_stats(model, x)
            for k in all_norms:
                if n_stats[k].size > 0:
                    all_norms[k].append(n_stats[k])
            fro_vals.append(patch_attention_frobenius(model, x))
            reg_mass_vals.append(register_attention_mass(model, x))
        patch_arr = np.concatenate(all_norms["patch"])
        spec_arr = np.concatenate(all_norms["special"]) if all_norms["special"] else np.array([])
        reg_arr = np.concatenate(all_norms["register"]) if all_norms["register"] else np.array([])

        med = float(np.median(patch_arr))
        iqr = float(np.percentile(patch_arr, 75) - np.percentile(patch_arr, 25))
        cutoff = med + ARTIFACT_THRESHOLD_IQR * iqr
        outlier_pct = float(np.mean(patch_arr > cutoff) * 100)

        diag_results[f"n={n_reg}"] = {
            "patch_norm_median": med,
            "patch_norm_iqr": iqr,
            "patch_norm_p99": float(np.percentile(patch_arr, 99)),
            "patch_norm_max": float(patch_arr.max()),
            "patch_norm_mean": float(patch_arr.mean()),
            "patch_norm_std": float(patch_arr.std()),
            "outlier_cutoff": cutoff,
            "outlier_pct": outlier_pct,
            "attn_frobenius_mean": float(np.mean(fro_vals)),
            "attn_frobenius_std": float(np.std(fro_vals)),
            "register_attn_mass_mean": float(np.mean(reg_mass_vals)),
            "register_attn_mass_std": float(np.std(reg_mass_vals)),
            "n_special": int(spec_arr.size),
            "n_register": int(reg_arr.size),
            "n_patch": int(patch_arr.size),
            "register_norm_mean": float(reg_arr.mean()) if reg_arr.size else None,
            "special_norm_mean": float(spec_arr.mean()) if spec_arr.size else None,
        }
        norm_dump[f"patch_norms_n{n_reg}"] = patch_arr.astype(np.float32)
        if reg_arr.size:
            norm_dump[f"register_norms_n{n_reg}"] = reg_arr.astype(np.float32)
        norm_dump[f"special_norms_n{n_reg}"] = spec_arr.astype(np.float32)
        print(
            f"    n={n_reg}: patch_mean={diag_results[f'n={n_reg}']['patch_norm_mean']:.3f},"
            f" outlier_pct={outlier_pct:.2f}%,"
            f" attn_frob={diag_results[f'n={n_reg}']['attn_frobenius_mean']:.3f},"
            f" reg_mass={diag_results[f'n={n_reg}']['register_attn_mass_mean']:.4f}"
        )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez(os.path.join(ASSETS, "real_norms.npz"), **norm_dump)

    # ------------------------------------------------------------------
    # 2. Save attention maps for n=0 and n=4 for the same input.
    # ------------------------------------------------------------------
    print("[measure_real_results] capturing attention maps for n=0 and n=4 ...")
    set_seed()
    one_loader = DataLoader(Subset(ds, [0]), batch_size=1, collate_fn=collate)
    x_one, _ = next(iter(one_loader))
    x_one = x_one.to(DEVICE)

    set_seed()
    base_attn_model = ModifiedAST(MODEL_ID, NUM_CLASSES, n_registers=0).to(DEVICE).eval()
    A_base = collect_attention_map(base_attn_model, x_one)
    del base_attn_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    set_seed()
    mod_attn_model = ModifiedAST(MODEL_ID, NUM_CLASSES, n_registers=4).to(DEVICE).eval()
    A_mod = collect_attention_map(mod_attn_model, x_one)
    del mod_attn_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    np.savez(
        os.path.join(ASSETS, "real_attention_maps.npz"),
        A_base=A_base.astype(np.float32),
        A_mod=A_mod.astype(np.float32),
    )

    # ------------------------------------------------------------------
    # 3. Mini fine-tuning comparison on one fold.
    # ------------------------------------------------------------------
    print("[measure_real_results] mini fine-tuning on fold=5 (smaller subset) ...")
    EPOCHS = 2
    BATCH = 4
    LR = 1e-4
    TRAIN_PER_FOLD = 80
    TEST_PER_FOLD = 32

    train_idx_all = [i for f, idxs in fold_to_idx.items() if f != 5 for i in idxs]
    test_idx_all = fold_to_idx[5]
    rng = random.Random(SEED)
    train_idx = rng.sample(train_idx_all, TRAIN_PER_FOLD)
    test_idx = rng.sample(test_idx_all, TEST_PER_FOLD)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH, shuffle=False, collate_fn=collate)

    training_results: Dict[str, Dict] = {}
    per_class_dump: Dict[str, np.ndarray] = {}
    training_dynamics: Dict[str, Dict[str, List[float]]] = {}

    for n_reg in [0, 4]:
        print(f"  > training n_registers={n_reg}")
        set_seed()
        model = ModifiedAST(MODEL_ID, NUM_CLASSES, n_registers=n_reg).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        history = {"train_acc": [], "test_acc": [], "attn_frob": [], "reg_mass": []}
        t0 = time.time()
        ep_preds = ep_targets = []
        for ep in range(EPOCHS):
            model.train()
            correct = total = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                logits = model(x)["logits"]
                loss = crit(logits, y)
                loss.backward()
                opt.step()
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
            train_acc = correct / max(total, 1)

            model.eval()
            correct = total = 0
            ep_preds, ep_targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)["logits"]
                    preds = logits.argmax(-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    ep_preds.append(preds.cpu().numpy())
                    ep_targets.append(y.cpu().numpy())
            test_acc = correct / max(total, 1)

            with torch.no_grad():
                xb, _ = next(iter(test_loader))
                xb = xb.to(DEVICE)
                fro = patch_attention_frobenius(model, xb)
                rm = register_attention_mass(model, xb)
            history["train_acc"].append(float(train_acc))
            history["test_acc"].append(float(test_acc))
            history["attn_frob"].append(float(fro))
            history["reg_mass"].append(float(rm))
            print(
                f"    n={n_reg} ep={ep+1}/{EPOCHS}: train_acc={train_acc:.3f},"
                f" test_acc={test_acc:.3f}, frob={fro:.3f}, reg_mass={rm:.4f}"
            )

        elapsed = time.time() - t0
        training_results[f"n={n_reg}"] = {
            "best_train_acc": max(history["train_acc"]),
            "best_test_acc": max(history["test_acc"]),
            "final_test_acc": history["test_acc"][-1],
            "final_attn_frob": history["attn_frob"][-1],
            "final_reg_mass": history["reg_mass"][-1],
            "epochs": EPOCHS,
            "train_subset": TRAIN_PER_FOLD,
            "test_subset": TEST_PER_FOLD,
            "wall_time_sec": elapsed,
        }
        training_dynamics[f"n={n_reg}"] = history

        per_class_dump[f"preds_n{n_reg}"] = np.concatenate(ep_preds)
        per_class_dump[f"targets_n{n_reg}"] = np.concatenate(ep_targets)

        del model, opt
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez(os.path.join(ASSETS, "real_perclass.npz"), **per_class_dump)

    # ------------------------------------------------------------------
    # Write everything to JSON.
    # ------------------------------------------------------------------
    summary = {
        "device": str(DEVICE),
        "model_id": MODEL_ID,
        "seed": SEED,
        "diag_n_samples_used": 64,
        "diagnostic": diag_results,
        "training_protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": LR,
            "train_per_fold": TRAIN_PER_FOLD,
            "test_per_fold": TEST_PER_FOLD,
            "test_fold": 5,
            "note": "Reduced compute due to MPS / time budget; not full 5-fold CV.",
        },
        "training": training_results,
        "training_dynamics": training_dynamics,
        "label_names": label_names,
        "artifact_threshold_iqr_multiplier": ARTIFACT_THRESHOLD_IQR,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[measure_real_results] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
