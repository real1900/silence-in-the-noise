"""Layer-wise hidden-state L2 norm analysis on the AudioSet-pretrained AST.

Computes mean / max / outlier-percent of patch-token L2 norms at every encoder
layer (1-12), for n_reg in {0, 4}. Strengthens the "AST has no artifact phase
transition" claim by showing the property holds at every depth, not just the
final layer.

Output:
  paper_assets/layer_wise_norms.json
  paper_assets/fig_layerwise_norms.pdf
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import ASTModel, ASTFeatureExtractor

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    from datasets import load_dataset, Audio as HFAudio
    HAS_HF = True
except ImportError:
    HAS_HF = False

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "paper_assets")
os.makedirs(ASSETS, exist_ok=True)

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
N_UTTS = 64
OUTLIER_IQR_MULT = 2.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class HookedAST(nn.Module):
    """AST that captures the post-LayerNorm hidden state of every encoder layer."""
    def __init__(self, n_registers: int = 0):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(MODEL_ID, attn_implementation="eager")
        self.embed_dim = self.encoder.config.hidden_size
        self.n_registers = n_registers
        if n_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, n_registers, self.embed_dim))
            nn.init.normal_(self.register_tokens, std=0.02)

        self._layer_outs: list[torch.Tensor] = []
        for layer in self.encoder.encoder.layer:
            layer.register_forward_hook(self._capture_layer)

    def _capture_layer(self, module, inp, out):
        if isinstance(out, tuple):
            self._layer_outs.append(out[0].detach())
        else:
            self._layer_outs.append(out.detach())

    def forward(self, input_values: torch.Tensor):
        self._layer_outs = []
        emb = self.encoder.embeddings(input_values)
        if self.n_registers > 0:
            B = emb.shape[0]
            cls_dst = emb[:, :2, :]
            patches = emb[:, 2:, :]
            regs = self.register_tokens.expand(B, -1, -1)
            emb = torch.cat([cls_dst, regs, patches], dim=1)
        self.encoder.encoder(emb)
        return self._layer_outs

    def patch_offset(self):
        return 2 + self.n_registers


# ----------------------------------------------------------------------
# ESC-50 direct loader (mirrors train_full_5fold.py)
# ----------------------------------------------------------------------
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_DIR = os.path.join(ROOT, "esc50_data")


def load_esc50_paths():
    audio_dir = os.path.join(ESC50_DIR, "ESC-50-master", "audio")
    meta_csv = os.path.join(ESC50_DIR, "ESC-50-master", "meta", "esc50.csv")
    if not os.path.exists(meta_csv):
        os.makedirs(ESC50_DIR, exist_ok=True)
        zip_path = os.path.join(ESC50_DIR, "esc50.zip")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ESC50_DIR)
        os.remove(zip_path)
    rows = []
    with open(meta_csv) as f:
        for r in csv.DictReader(f):
            rows.append(os.path.join(audio_dir, r["filename"]))
    return rows


def load_audio_arrays(n_utts: int):
    """Return list of np.float32 audio arrays. Tries the local ESC-50 zip first
    (Linux cluster path), then falls back to the HF dataset cache (macOS path
    where torchcodec already works)."""
    # 1. direct download path (works on linux + on macos if SSL is set up)
    try:
        paths = load_esc50_paths()[:n_utts]
        if sf is None:
            raise RuntimeError("soundfile not installed")
        out = []
        for p in paths:
            a, _ = sf.read(p, dtype="float32", always_2d=False)
            if a.ndim > 1:
                a = a.mean(axis=-1)
            out.append(a)
        return out
    except Exception as e:
        print(f"[layer_wise_norms] direct ESC-50 load failed ({e}); "
              f"trying HuggingFace cache fallback")
    # 2. HF dataset cache (uses torchcodec on macOS, just works)
    if not HAS_HF:
        raise RuntimeError("HuggingFace datasets not installed; cannot fall back")
    ds = load_dataset("ashraq/esc50", split=f"train[:{n_utts}]")
    ds = ds.cast_column("audio", HFAudio(sampling_rate=SAMPLE_RATE))
    out = []
    for row in ds:
        a = row["audio"]
        if hasattr(a, "get_all_samples"):
            samples = a.get_all_samples()
            arr = samples.data.cpu().numpy().squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            out.append(arr.astype(np.float32))
        elif isinstance(a, dict) and "array" in a:
            arr = np.asarray(a["array"], dtype=np.float32)
            out.append(arr if arr.ndim == 1 else arr.mean(-1))
        else:
            raise RuntimeError(f"unexpected HF audio type: {type(a)}")
    return out


def main():
    print(f"[layer_wise_norms] device={DEVICE}")
    arrays = load_audio_arrays(N_UTTS)
    print(f"[layer_wise_norms] loaded {len(arrays)} audio clips")
    fx = ASTFeatureExtractor()

    summary = {}
    for n_reg in [0, 4]:
        print(f"\n--- n_reg={n_reg} ---")
        torch.manual_seed(42)
        model = HookedAST(n_registers=n_reg).to(DEVICE).eval()
        # Per-layer arrays of patch norms aggregated across utterances
        layers_patch_norms: list[list[np.ndarray]] = [[] for _ in range(12)]
        with torch.no_grad():
            for i in range(0, len(arrays), 4):
                batch = arrays[i:i + 4]
                feats = fx(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                x = feats["input_values"].to(DEVICE)
                outs = model(x)  # list of 12 (B, S, d)
                p_off = model.patch_offset()
                for li, h in enumerate(outs):
                    norms = h.norm(dim=-1).cpu().numpy()
                    layers_patch_norms[li].append(norms[:, p_off:].reshape(-1))

        layer_stats = []
        for li, lst in enumerate(layers_patch_norms):
            arr = np.concatenate(lst)
            med = float(np.median(arr))
            iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            cutoff = med + OUTLIER_IQR_MULT * iqr
            outlier_pct = float(np.mean(arr > cutoff) * 100)
            layer_stats.append({
                "layer": li + 1,
                "patch_norm_median": med,
                "patch_norm_max": float(arr.max()),
                "patch_norm_iqr": iqr,
                "outlier_pct": outlier_pct,
                "max_to_median": float(arr.max() / med),
            })
            print(f"  layer {li+1:>2}: med={med:.2f} max={arr.max():.2f} "
                  f"max/med={arr.max()/med:.2f}  outlier%={outlier_pct:.2f}")
        summary[f"n={n_reg}"] = layer_stats
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # save JSON
    with open(os.path.join(ASSETS, "layer_wise_norms.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {os.path.join(ASSETS, 'layer_wise_norms.json')}")

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, key in enumerate(["max_to_median", "outlier_pct"]):
        for n_reg, color in [(0, "#1565C0"), (4, "#C62828")]:
            stats = summary[f"n={n_reg}"]
            xs = [s["layer"] for s in stats]
            ys = [s[key] for s in stats]
            ax[i].plot(xs, ys, "o-", color=color, linewidth=1.5, label=f"n={n_reg}")
        ax[i].set_xlabel("Encoder layer")
        ax[i].grid(alpha=0.3)
        ax[i].legend()
    ax[0].set_ylabel("max / median patch L2 norm")
    ax[0].set_title("(a) Outlier extremity per layer")
    ax[1].set_ylabel("Patch outlier rate (%)  [median + 2.5x IQR]")
    ax[1].set_title("(b) Outlier rate per layer")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, "fig_layerwise_norms.pdf"))
    plt.close(fig)
    print(f"Wrote {os.path.join(ASSETS, 'fig_layerwise_norms.pdf')}")


if __name__ == "__main__":
    main()
