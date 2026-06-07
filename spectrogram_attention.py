"""Visualize attention overlaid on the input spectrogram.

For one ESC-50 utterance, render:
  (a) the log-Mel spectrogram
  (b) the spatial attention map (CLS->patches) of the baseline AST
  (c) the same for the AST+Reg model
  (d) attention received BY the register tokens (where they look)

Output: paper_assets/fig_spec_attention.pdf
"""
from __future__ import annotations

import csv
import os
import sys
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import ASTModel, ASTFeatureExtractor

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "paper_assets")
os.makedirs(ASSETS, exist_ok=True)

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class HookedAST(nn.Module):
    def __init__(self, n_registers: int = 0):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(MODEL_ID, attn_implementation="eager")
        self.embed_dim = self.encoder.config.hidden_size
        self.n_registers = n_registers
        if n_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_registers, self.embed_dim))
            nn.init.normal_(self.register_tokens, std=0.02)
        self._last_attn = None
        self.encoder.encoder.layer[-1].attention.attention.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._last_attn = out[1].detach() if isinstance(out, tuple) else None

    def forward(self, x):
        self._last_attn = None
        emb = self.encoder.embeddings(x)
        if self.n_registers > 0:
            B = emb.shape[0]
            cls_dst = emb[:, :2, :]
            patches = emb[:, 2:, :]
            regs = self.register_tokens.expand(B, -1, -1)
            emb = torch.cat([cls_dst, regs, patches], dim=1)
        self.encoder.encoder(emb)
        return self._last_attn

    def patch_offset(self):
        return 2 + self.n_registers


ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_DIR = os.path.join(ROOT, "esc50_data")


def find_esc50_audio(idx: int = 0):
    audio_dir = os.path.join(ESC50_DIR, "ESC-50-master", "audio")
    meta_csv = os.path.join(ESC50_DIR, "ESC-50-master", "meta", "esc50.csv")
    if os.path.exists(meta_csv):
        with open(meta_csv) as f:
            rows = list(csv.DictReader(f))
        r = rows[idx]
        return os.path.join(audio_dir, r["filename"]), r["category"]
    # try HF cache (macOS works)
    try:
        from datasets import load_dataset, Audio as HFAudio
        ds = load_dataset("ashraq/esc50", split=f"train[{idx}:{idx+1}]")
        ds = ds.cast_column("audio", HFAudio(sampling_rate=SAMPLE_RATE))
        row = ds[0]
        a = row["audio"]
        if hasattr(a, "get_all_samples"):
            samples = a.get_all_samples()
            arr = samples.data.cpu().numpy().squeeze().astype(np.float32)
        else:
            arr = np.asarray(a["array"], dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(-1)
        # save to a temp wav so the rest of the script keeps a "path" interface
        os.makedirs(ESC50_DIR, exist_ok=True)
        out_path = os.path.join(ESC50_DIR, f"hf_sample_{idx}.wav")
        sf.write(out_path, arr, SAMPLE_RATE)
        return out_path, row["category"]
    except Exception:
        pass
    # last resort: direct download
    os.makedirs(ESC50_DIR, exist_ok=True)
    zp = os.path.join(ESC50_DIR, "esc50.zip")
    urllib.request.urlretrieve(ESC50_URL, zp)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(ESC50_DIR)
    os.remove(zp)
    with open(meta_csv) as f:
        rows = list(csv.DictReader(f))
    r = rows[idx]
    return os.path.join(audio_dir, r["filename"]), r["category"]


def main():
    print(f"[spec_attention] device={DEVICE}")
    path, category = find_esc50_audio(0)
    print(f"[spec_attention] sample: {os.path.basename(path)}  ({category})")

    arr, _ = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(-1)

    fx = ASTFeatureExtractor()
    feats = fx([arr], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    x = feats["input_values"].to(DEVICE)  # (1, 1024, 128)

    spec = feats["input_values"][0].numpy().T  # (128, 1024)

    # AST patch grid: 16x16 patch with stride 10 over (1024, 128)
    # Output grid is approximately ((1024-16)/10+1, (128-16)/10+1) = (101, 12)
    GRID_T, GRID_F = 101, 12

    def cls_to_grid(attn_row, p_off):
        """Take the [CLS]-row of attn (length S), slice off patches, reshape to (T, F)."""
        # attn_row over keys; we want [CLS] -> patches sub-row
        patches_attn = attn_row[p_off:]
        # Reshape; AST patches are time-major (T outer, F inner)
        if patches_attn.size != GRID_T * GRID_F:
            # for non-trivial register counts the patch length may differ from
            # the canonical 1212; recompute grid from token count
            T = (1024 - 16) // 10 + 1
            F = (128 - 16) // 10 + 1
            assert patches_attn.size == T * F, (patches_attn.size, T, F)
            return patches_attn.reshape(T, F).T
        return patches_attn.reshape(GRID_T, GRID_F).T

    results = {}
    for n_reg in [0, 4]:
        torch.manual_seed(42)
        model = HookedAST(n_registers=n_reg).to(DEVICE).eval()
        with torch.no_grad():
            attn = model(x)  # (1, H, S, S)
        a = attn[0].mean(0).cpu().numpy()  # head-averaged (S, S)
        p_off = model.patch_offset()
        cls_to_p = cls_to_grid(a[0], p_off)
        results[n_reg] = {
            "attn_grid_cls": cls_to_p,
            "attn_full": a,
            "p_off": p_off,
        }
        if n_reg > 0:
            # Where do registers look? Attention rows for register slots
            reg_rows = a[2:2 + n_reg, :]  # (n_reg, S)
            # Average register attention over registers
            mean_reg_row = reg_rows.mean(0)
            results[n_reg]["reg_to_p_grid"] = cls_to_grid(mean_reg_row, p_off)
        del model

    # ---- plot ----
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    im0 = axes[0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"(a) Mel spectrogram\n{category}")
    axes[0].set_xlabel("Time frame"); axes[0].set_ylabel("Mel bin")
    plt.colorbar(im0, ax=axes[0], fraction=0.025)

    im1 = axes[1].imshow(results[0]["attn_grid_cls"], aspect="auto", origin="lower",
                          cmap="viridis")
    axes[1].set_title("(b) Baseline [CLS]→patch attention\n(spatial layout)")
    axes[1].set_xlabel("Patch time idx"); axes[1].set_ylabel("Patch freq idx")
    plt.colorbar(im1, ax=axes[1], fraction=0.025)

    im2 = axes[2].imshow(results[4]["attn_grid_cls"], aspect="auto", origin="lower",
                          cmap="viridis", vmax=results[0]["attn_grid_cls"].max())
    axes[2].set_title("(c) AST+Reg [CLS]→patch attention\n(remaining patch attn)")
    axes[2].set_xlabel("Patch time idx")
    plt.colorbar(im2, ax=axes[2], fraction=0.025)

    im3 = axes[3].imshow(results[4]["reg_to_p_grid"], aspect="auto", origin="lower",
                          cmap="plasma")
    axes[3].set_title("(d) Register→patch attention\n(what registers absorb)")
    axes[3].set_xlabel("Patch time idx")
    plt.colorbar(im3, ax=axes[3], fraction=0.025)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "fig_spec_attention.pdf"))
    plt.close(fig)
    print(f"Wrote {os.path.join(ASSETS, 'fig_spec_attention.pdf')}")


if __name__ == "__main__":
    main()
