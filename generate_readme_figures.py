"""Generate README hero visualizations from the paper's real measurements.

Outputs (to assets/):
  fig_readme_maxmedian.png  - max/median norm ratio across transformer families
  fig_readme_int8.png       - PT-INT8 fidelity + size reduction across backends

Numbers: AST values are this work's measurements; other families' max/median are
the cross-study, illustrative values from the paper's comparison table (cited).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "paper_assets")
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

JHU_BLUE = "#002D72"
SAFE = "#1b9e77"      # green
DANGER = "#d62728"    # red
GREY = "#9aa0a6"

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------------------------------------------------------------------
# Figure 1: max-to-median norm ratio across transformer families (the diagnostic)
# ----------------------------------------------------------------------------
families = [
    ("AST-Base (this work)", 1.23, True),
    ("ViT-Tiny", 5.0, False),
    ("BERT-Base", 6.0, False),
    ("ViT-L / DINOv2", 10.0, False),
    ("LLaMA-7B", 50.0, False),
    ("OPT-175B", 100.0, False),
]
families = sorted(families, key=lambda x: x[1])
labels = [f[0] for f in families]
vals = [f[1] for f in families]
is_ast = [f[2] for f in families]
colors = [SAFE if a else (GREY if v < 5 else DANGER) for v, a in zip(vals, is_ast)]

fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=200)
y = range(len(vals))
ax.barh(list(y), vals, color=colors, height=0.62, zorder=3)
ax.set_xscale("log")
ax.set_xlim(1, 160)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.set_xlabel("max-to-median final-layer token-norm ratio  $r$  (log scale)")
ax.set_title("One number predicts INT8-readiness: AST sits in the safe zone",
             fontweight="bold", color=JHU_BLUE, pad=12)

# safe / fix-needed zones
ax.axvspan(1, 2, color=SAFE, alpha=0.08, zorder=0)
ax.axvline(2, color=SAFE, ls="--", lw=1.3, zorder=2)
ax.axvline(5, color=DANGER, ls="--", lw=1.3, zorder=2)
ax.text(1.42, len(vals) - 0.35, "INT8-safe\n$r<2$", color=SAFE, ha="center",
        va="top", fontsize=10, fontweight="bold")
ax.text(9, len(vals) - 0.35, "needs outlier fix\n$r>5$", color=DANGER, ha="center",
        va="top", fontsize=10, fontweight="bold")

for i, (v, a) in enumerate(zip(vals, is_ast)):
    tag = "  no fix needed" if a else "  fix required"
    ax.text(v * 1.06, i, f"{v:g}{tag if (a or v>5) else ''}",
            va="center", ha="left", fontsize=10,
            fontweight="bold" if a else "normal",
            color=JHU_BLUE if a else "#444")
ax.text(0.99, -0.16, "Other families: illustrative cross-study values (Bondarenko 2023; Sun 2024; Darcet 2024). AST: this work.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#777")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_readme_maxmedian.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_readme_maxmedian.png")

# ----------------------------------------------------------------------------
# Figure 2: PT-INT8 fidelity + size reduction across backends (real measurements)
# ----------------------------------------------------------------------------
cpu = json.load(open(os.path.join(ASSETS, "quantization_results.json")))   # M1/qnnpack
cuda = json.load(open(os.path.join(ASSETS, "cuda_quantization_results.json")))
# x86 values are reported in the paper's Table III
backends = ["x86\n(fbgemm)", "Apple M1\n(qnnpack)", "RTX A4000\n(bitsandbytes)"]
cosine = [0.980, round(cpu["token_cosine_similarity_mean"], 4), cuda["token_cosine_mean"]]
top5 = [0.730, round(cpu["top5_rank_agreement_cls"], 3), cuda["top5_rank_agreement"]]
size = [74, round(cpu["size_reduction_pct"]), round(cuda["size_reduction_pct"])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200,
                               gridspec_kw={"width_ratios": [1.5, 1]})
x = range(len(backends))
w = 0.38
b1 = ax1.bar([i - w/2 for i in x], cosine, w, label="token-state cosine vs FP32/FP16",
             color=JHU_BLUE, zorder=3)
b2 = ax1.bar([i + w/2 for i in x], top5, w, label="top-5 prediction agreement",
             color="#5b8def", zorder=3)
ax1.axhline(0.99, color=SAFE, ls="--", lw=1.1)
ax1.text(2.45, 0.992, "cosine bar 0.99", color=SAFE, fontsize=8, ha="right")
ax1.axhline(0.90, color="#e0a800", ls="--", lw=1.1)
ax1.text(2.45, 0.872, "top-5 bar 0.90", color="#b8860b", fontsize=8, ha="right")
ax1.set_ylim(0.6, 1.02)
ax1.set_xticks(list(x)); ax1.set_xticklabels(backends)
ax1.set_ylabel("fidelity (higher is better)")
ax1.set_title("Naive PT-INT8 fidelity — no outlier fixes",
              fontweight="bold", color=JHU_BLUE)
for bars in (b1, b2):
    for b in bars:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.004,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)

bars = ax2.bar(list(x), size, color=SAFE, width=0.6, zorder=3)
ax2.set_ylim(0, 100)
ax2.set_xticks(list(x)); ax2.set_xticklabels(backends)
ax2.set_ylabel("model-size reduction (%)")
ax2.set_title("Smaller, for free", fontweight="bold", color=JHU_BLUE)
for b in bars:
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
             f"{int(b.get_height())}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=JHU_BLUE)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_readme_int8.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig_readme_int8.png")
