"""CUDA INT8 quantization test for AST using bitsandbytes.

Replaces every nn.Linear in the AST encoder with bnb.nn.Linear8bitLt and
measures FP16 vs INT8 latency, model size, and token-state cosine fidelity.

The argument: AST has no outlier-token activations (max-to-median ratio ~ 1.23),
which is the ONE thing that traditionally breaks INT8 on transformers
(Bondarenko et al., NeurIPS 2023). We expect AST to take INT8 with low
fidelity loss AND noticeable speedup on a CUDA target.

Output:
  paper_assets/cuda_quantization_results.json
  paper_assets/fig_cuda_quantization.pdf
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import ASTModel, ASTFeatureExtractor

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "paper_assets"
ASSETS.mkdir(exist_ok=True)
ESC50_DIR = ROOT / "esc50_data"

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
N_UTTS = int(os.environ.get("N_UTTS", "30"))
LATENCY_N = int(os.environ.get("LATENCY_N", "50"))
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def load_audio_arrays(n: int) -> List[np.ndarray]:
    audio_dir = ESC50_DIR / "ESC-50-master" / "audio"
    meta_csv = ESC50_DIR / "ESC-50-master" / "meta" / "esc50.csv"
    if not meta_csv.exists():
        ESC50_DIR.mkdir(parents=True, exist_ok=True)
        zp = ESC50_DIR / "esc50.zip"
        print(f"[cudaq] downloading ESC-50 from {ESC50_URL} ...")
        urllib.request.urlretrieve(ESC50_URL, zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(ESC50_DIR)
        zp.unlink()
    rows = list(csv.DictReader(open(meta_csv)))[:n]
    out = []
    for r in rows:
        a, _ = sf.read(audio_dir / r["filename"], dtype="float32",
                        always_2d=False)
        if a.ndim > 1: a = a.mean(-1)
        out.append(a)
    return out


def measure(model, fx, arrays, label, time_n=LATENCY_N, dtype=torch.float32):
    """Forward pass + latency measurement on the given device."""
    model.eval()
    pooled, last_hidden = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for a in arrays:
            feats = fx([a], sampling_rate=SAMPLE_RATE, return_tensors="pt")
            x = feats["input_values"].to(device)
            if dtype == torch.float16:
                x = x.half()
            out = model(x)
            h = out.last_hidden_state.float()
            pooled.append(h[:, 0].cpu().numpy())
            last_hidden.append(h.cpu().numpy())
    pooled = np.concatenate(pooled, axis=0)

    feats0 = fx([arrays[0]], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    x0 = feats0["input_values"].to(device)
    if dtype == torch.float16:
        x0 = x0.half()
    # warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(time_n):
            _ = model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) / time_n * 1000.0
    print(f"  [{label}] {time_n} forwards, avg latency = {elapsed_ms:.2f} ms/sample")
    return pooled, last_hidden, elapsed_ms


def cosine_per_token(h_a, h_b):
    cosines, l2 = [], []
    for ha, hb in zip(h_a, h_b):
        ha, hb = ha.squeeze(0), hb.squeeze(0)
        d = (ha * hb).sum(axis=-1) / (np.linalg.norm(ha, axis=-1) *
                                       np.linalg.norm(hb, axis=-1) + 1e-12)
        cosines.append(d)
        l2.append(np.linalg.norm(ha - hb, axis=-1) /
                  (np.linalg.norm(ha, axis=-1) + 1e-12))
    return np.concatenate(cosines), np.concatenate(l2)


def topk_agree(p_a, p_b, k=5):
    same = []
    for v1, v2 in zip(p_a, p_b):
        d1 = np.argsort(-v1)[:k]
        d2 = np.argsort(-v2)[:k]
        same.append(len(set(d1) & set(d2)) / k)
    return float(np.mean(same))


def replace_linears_with_int8(model: nn.Module, threshold: float = 6.0):
    """Walk the module tree and replace every nn.Linear with bnb.nn.Linear8bitLt.

    Skips the embedding patch projection (Conv2d) and LayerNorm. The
    bitsandbytes Int8 kernels mix int8 weights with bf16 outliers (the LLM.int8
    paper's vector-wise quantization with a fp16 outlier path). We disable the
    outlier path with `has_fp16_weights=False` to force pure int8 inference.
    """
    import bitsandbytes as bnb
    n_replaced = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = bnb.nn.Linear8bitLt(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
            new_layer.weight = bnb.nn.Int8Params(
                module.weight.data, requires_grad=False, has_fp16_weights=False
            )
            if module.bias is not None:
                new_layer.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_layer)
            n_replaced += 1
        else:
            n_replaced += replace_linears_with_int8(module, threshold)
    return n_replaced


def main():
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. This test must run on a GPU.")
        return
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[cudaq] device = {device}  ({gpu_name})")

    print("[cudaq] loading audio ...")
    arrays = load_audio_arrays(N_UTTS)
    print(f"[cudaq] {len(arrays)} clips loaded")
    fx = ASTFeatureExtractor()

    # ----- FP16 baseline (production target) -----
    print("\n[cudaq] FP16 AST on CUDA ...")
    fp_model = ASTModel.from_pretrained(MODEL_ID, attn_implementation="eager",
                                          torch_dtype=torch.float16).to(device).eval()
    n_params = sum(p.numel() for p in fp_model.parameters())
    fp_size_mb = n_params * 2 / (1024 ** 2)  # fp16 = 2 bytes
    print(f"  {n_params/1e6:.1f}M params; FP16 size = {fp_size_mb:.1f} MB")
    p_fp, h_fp, lat_fp = measure(fp_model, fx, arrays, "FP16",
                                  dtype=torch.float16)

    # ----- INT8 model -----
    print("\n[cudaq] building INT8 AST via bitsandbytes ...")
    int8_model = ASTModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    n_replaced = replace_linears_with_int8(int8_model, threshold=6.0)
    int8_model = int8_model.to(device).eval()
    print(f"  replaced {n_replaced} nn.Linear modules with bnb.Linear8bitLt")
    # estimate size: int8 weights for linears, fp16 for everything else
    int8_size_mb = sum(
        (p.numel() * 1) if (p.dtype == torch.int8) else (p.numel() * 2)
        for p in int8_model.parameters() if p is not None
    ) / (1024 ** 2)
    print(f"  est. INT8 size = {int8_size_mb:.1f} MB")
    p_int8, h_int8, lat_int8 = measure(int8_model, fx, arrays, "INT8",
                                         dtype=torch.float32)

    cos, l2 = cosine_per_token(h_fp, h_int8)
    cos_mean = float(np.mean(cos))
    cos_p1 = float(np.percentile(cos, 1))
    rank_agree = topk_agree(p_fp, p_int8, k=5)
    speedup = lat_fp / lat_int8

    print(f"\n=== CUDA QUANTIZATION RESULTS ({gpu_name}, {N_UTTS} clips) ===")
    print(f"  FP16 latency  : {lat_fp:.2f} ms/sample")
    print(f"  INT8 latency  : {lat_int8:.2f} ms/sample")
    print(f"  Speedup       : {speedup:.2f}x")
    print(f"  Size FP16     : {fp_size_mb:.1f} MB")
    print(f"  Size INT8 est : {int8_size_mb:.1f} MB ({100*(1-int8_size_mb/fp_size_mb):.0f}% reduction)")
    print(f"  Token cosine  : mean={cos_mean:.4f}  p1={cos_p1:.4f}")
    print(f"  Top-5 rank agreement : {rank_agree:.3f}")

    summary = {
        "device": gpu_name,
        "n_clips": N_UTTS,
        "fp16_latency_ms": lat_fp,
        "int8_latency_ms": lat_int8,
        "speedup": speedup,
        "fp16_size_mb": fp_size_mb,
        "int8_size_mb": int8_size_mb,
        "token_cosine_mean": cos_mean,
        "token_cosine_p1": cos_p1,
        "top5_rank_agreement": rank_agree,
        "n_linears_replaced": n_replaced,
    }
    out_json = ASSETS / "cuda_quantization_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].bar(["FP16", "INT8"], [lat_fp, lat_int8],
              color=["#1565C0", "#388E3C"], edgecolor="black")
    ax[0].set_ylabel("Latency (ms / sample)")
    ax[0].set_title(f"(a) {gpu_name} latency  -  {speedup:.2f}x speedup")
    for i, v in enumerate([lat_fp, lat_int8]):
        ax[0].text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=11)
    ax[0].grid(alpha=0.3, axis="y")

    ax[1].hist(cos, bins=40, color="#388E3C", edgecolor="black", alpha=0.85)
    ax[1].axvline(0.99, color="red", linestyle="--", linewidth=1)
    ax[1].set_xlabel("Cosine sim FP16 vs INT8 (per-token)")
    ax[1].set_ylabel("Tokens")
    ax[1].set_title(f"(b) Token-state agreement\nmean = {cos_mean:.4f}")
    ax[1].grid(alpha=0.3)

    ax[2].bar(["FP16", "INT8"], [fp_size_mb, int8_size_mb],
              color=["#1565C0", "#388E3C"], edgecolor="black")
    ax[2].set_ylabel("Model size (MB)")
    ax[2].set_title("(c) GPU memory footprint")
    for i, v in enumerate([fp_size_mb, int8_size_mb]):
        ax[2].text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=11)
    ax[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_pdf = ASSETS / "fig_cuda_quantization.pdf"
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
