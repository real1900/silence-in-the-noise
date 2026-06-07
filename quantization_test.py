"""Quantization-friendliness experiment: does AST's outlier-free profile mean
it tolerates INT8 quantization without the gating/clipping fixes that ViTs
typically require?

We compare FP32 vs INT8-dynamic-quantized AST on 100 ESC-50 clips. Metrics:
  - cosine similarity of penultimate-layer hidden states (FP32 vs INT8)
  - hidden-state Frobenius drift relative to FP32 norm
  - top-5 logit-rank agreement (proxy for prediction stability)
  - inference latency speedup on CPU (ms / sample)

Output:
  paper_assets/quantization_results.json
  paper_assets/fig_quantization.pdf
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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

# fbgemm is the fast x86 INT8 backend; qnnpack is the slower ARM/mobile fallback.
# Prefer fbgemm where available (x86 servers), otherwise use qnnpack (M1/Mx).
if "fbgemm" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "fbgemm"
elif "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"
print(f"[quant] backend = {torch.backends.quantized.engine}")
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
LATENCY_N = int(os.environ.get("LATENCY_N", "30"))
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def load_audio_arrays(n: int) -> List[np.ndarray]:
    audio_dir = ESC50_DIR / "ESC-50-master" / "audio"
    meta_csv = ESC50_DIR / "ESC-50-master" / "meta" / "esc50.csv"
    if not meta_csv.exists():
        # Direct GitHub download (works on Linux/CUDA where torchcodec is broken)
        try:
            ESC50_DIR.mkdir(parents=True, exist_ok=True)
            zp = ESC50_DIR / "esc50.zip"
            print(f"[quant] downloading ESC-50 from {ESC50_URL} ...")
            urllib.request.urlretrieve(ESC50_URL, zp)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(ESC50_DIR)
            zp.unlink()
        except Exception as e:
            print(f"[quant] direct download failed: {e}")
    if meta_csv.exists():
        rows = list(csv.DictReader(open(meta_csv)))[:n]
        out = []
        for r in rows:
            a, _ = sf.read(audio_dir / r["filename"], dtype="float32",
                            always_2d=False)
            if a.ndim > 1: a = a.mean(-1)
            out.append(a)
        return out
    # fallback: HF datasets (works on macOS where torchcodec is happy)
    from datasets import load_dataset, Audio
    ds = load_dataset("ashraq/esc50", split=f"train[:{n}]")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    out = []
    for row in ds:
        a = row["audio"]
        if hasattr(a, "get_all_samples"):
            arr = a.get_all_samples().data.cpu().numpy().squeeze()
        else:
            arr = np.asarray(a["array"], dtype=np.float32)
        if arr.ndim > 1: arr = arr.mean(-1)
        out.append(arr.astype(np.float32))
    return out


def measure_one(model, fx, arrays, label, time_n=None):
    if time_n is None:
        time_n = LATENCY_N
    """Forward all utterances; return last-hidden tensor (N, S, d) and per-clip
    pooled CLS vector (N, d) and inference-latency-ms list."""
    model.eval()
    pooled, last_hidden = [], []
    with torch.no_grad():
        for a in arrays:
            feats = fx([a], sampling_rate=SAMPLE_RATE, return_tensors="pt")
            x = feats["input_values"]
            out = model(x)
            h = out.last_hidden_state
            pooled.append(h[:, 0].cpu().numpy())
            last_hidden.append(h.cpu().numpy())
    pooled = np.concatenate(pooled, axis=0)

    # Latency: time `time_n` forward passes on a fresh model state
    feats0 = fx([arrays[0]], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    x0 = feats0["input_values"]
    # warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = model(x0)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(time_n):
            _ = model(x0)
    elapsed_ms = (time.time() - t0) / time_n * 1000.0
    print(f"  [{label}] avg forward latency = {elapsed_ms:.1f} ms/sample")
    return pooled, last_hidden, elapsed_ms


def cosine_per_token(h_fp32, h_int8):
    """h_fp32, h_int8: lists of (1, S, d) numpy arrays. Returns one cosine
    per token across all clips, then averaged."""
    cosines, l2_drifts = [], []
    for hf, hi in zip(h_fp32, h_int8):
        hf, hi = hf.squeeze(0), hi.squeeze(0)
        d = (hf * hi).sum(axis=-1) / (np.linalg.norm(hf, axis=-1) *
                                       np.linalg.norm(hi, axis=-1) + 1e-12)
        cosines.append(d)
        l2_drifts.append(np.linalg.norm(hf - hi, axis=-1) /
                         (np.linalg.norm(hf, axis=-1) + 1e-12))
    return np.concatenate(cosines), np.concatenate(l2_drifts)


def topk_agreement(p_fp32, p_int8, k=5):
    """Treat the pooled CLS vectors themselves as 'logits'; measure top-k rank
    agreement under raw projection - a proxy for prediction stability when
    the same downstream classifier is applied."""
    same = []
    for v1, v2 in zip(p_fp32, p_int8):
        d1 = np.argsort(-v1)[:k]
        d2 = np.argsort(-v2)[:k]
        same.append(len(set(d1) & set(d2)) / k)
    return float(np.mean(same))


def main():
    print("[quant] loading audio ...")
    arrays = load_audio_arrays(N_UTTS)
    print(f"[quant] {len(arrays)} clips")
    fx = ASTFeatureExtractor()

    print("\n[quant] loading FP32 AST on CPU ...")
    fp_model = ASTModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    # force CPU (quantization requires CPU)
    fp_model = fp_model.to("cpu").eval()
    n_params = sum(p.numel() for p in fp_model.parameters())
    fp_size_mb = n_params * 4 / (1024 ** 2)
    print(f"  {n_params/1e6:.1f}M params; FP32 size = {fp_size_mb:.1f} MB")
    p_fp, h_fp, lat_fp = measure_one(fp_model, fx, arrays, "FP32")

    print("\n[quant] applying dynamic INT8 quantization to all Linear layers ...")
    q_model = torch.quantization.quantize_dynamic(
        fp_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    # measure size of the quantized model by serialising state_dict
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(q_model.state_dict(), tmp.name)
    q_size_mb = os.path.getsize(tmp.name) / (1024 ** 2)
    os.unlink(tmp.name)
    print(f"  INT8 dynamic-quantized model state_dict on disk: {q_size_mb:.1f} MB "
          f"({100 * q_size_mb / fp_size_mb:.1f}% of FP32)")
    p_q, h_q, lat_q = measure_one(q_model, fx, arrays, "INT8")

    cos, l2 = cosine_per_token(h_fp, h_q)
    cos_mean, cos_p1 = float(np.mean(cos)), float(np.percentile(cos, 1))
    l2_mean, l2_p99 = float(np.mean(l2)), float(np.percentile(l2, 99))
    rank_agree = topk_agreement(p_fp, p_q, k=5)

    speedup = lat_fp / lat_q

    print(f"\n=== QUANTIZATION RESULTS (AST on {N_UTTS} ESC-50 clips, CPU) ===")
    print(f"  FP32 forward latency  : {lat_fp:.1f} ms/sample")
    print(f"  INT8 forward latency  : {lat_q:.1f} ms/sample  ({speedup:.2f}x speedup)")
    print(f"  Model size            : {fp_size_mb:.1f} MB -> {q_size_mb:.1f} MB "
          f"({100*(1-q_size_mb/fp_size_mb):.0f}% reduction)")
    print(f"  Token cosine sim FP32 vs INT8 : mean={cos_mean:.4f}  p1={cos_p1:.4f}")
    print(f"  Relative L2 drift     : mean={l2_mean:.4f}  p99={l2_p99:.4f}")
    print(f"  Top-5 CLS-rank agreement      : {rank_agree:.3f}")

    summary = {
        "n_clips": N_UTTS,
        "model_id": MODEL_ID,
        "fp32_latency_ms_per_sample": lat_fp,
        "int8_latency_ms_per_sample": lat_q,
        "speedup_int8_over_fp32": speedup,
        "fp32_size_mb": fp_size_mb,
        "int8_size_mb_state_dict": q_size_mb,
        "size_reduction_pct": 100 * (1 - q_size_mb / fp_size_mb),
        "token_cosine_similarity_mean": cos_mean,
        "token_cosine_similarity_p1": cos_p1,
        "relative_l2_drift_mean": l2_mean,
        "relative_l2_drift_p99": l2_p99,
        "top5_rank_agreement_cls": rank_agree,
    }
    out_json = ASSETS / "quantization_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    # plot
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))

    # (a) latency bar chart with speedup callout
    ax[0].bar(["FP32", "INT8"], [lat_fp, lat_q],
              color=["#1565C0", "#388E3C"], edgecolor="black")
    ax[0].set_ylabel("Forward latency (ms / sample, CPU)")
    ax[0].set_title(f"(a) Inference latency  -  {speedup:.2f}x speedup")
    for i, v in enumerate([lat_fp, lat_q]):
        ax[0].text(i, v * 1.02, f"{v:.0f} ms", ha="center", fontsize=11)
    ax[0].grid(alpha=0.3, axis="y")

    # (b) token cosine similarity histogram
    ax[1].hist(cos, bins=40, color="#388E3C", edgecolor="black",
               alpha=0.8)
    ax[1].axvline(0.99, color="red", linestyle="--", linewidth=1)
    ax[1].set_xlabel("Cosine sim, FP32 vs INT8 token state")
    ax[1].set_ylabel("Tokens")
    ax[1].set_title(f"(b) Token-state agreement\nmean = {cos_mean:.4f}")
    ax[1].grid(alpha=0.3)

    # (c) size comparison
    ax[2].bar(["FP32", "INT8"], [fp_size_mb, q_size_mb],
              color=["#1565C0", "#388E3C"], edgecolor="black")
    ax[2].set_ylabel("Model size on disk (MB)")
    ax[2].set_title(f"(c) Disk footprint  -  "
                    f"{100*(1-q_size_mb/fp_size_mb):.0f}% reduction")
    for i, v in enumerate([fp_size_mb, q_size_mb]):
        ax[2].text(i, v * 1.02, f"{v:.0f} MB", ha="center", fontsize=11)
    ax[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_pdf = ASSETS / "fig_quantization.pdf"
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
