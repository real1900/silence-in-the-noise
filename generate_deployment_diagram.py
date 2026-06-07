"""Single-figure deployment diagram for the paper. Shows the AST training-time
to deployment-time pipeline, with INT8 quantization as the bridge that opens
multiple device tiers.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "paper_assets"
ASSETS.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")

# === Training side ===
def box(x, y, w, h, label, color, edge="black", textsize=9, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06",
        edgecolor=edge, facecolor=color, linewidth=1.1, linestyle=ls,
    ))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=textsize)


# Training stage
box(0.3, 3.0, 2.4, 1.4, "AudioSet\n(2M clips)", "#cde4ff")
box(3.0, 3.0, 2.4, 1.4, "AST-Base FP32\n86M params, 329 MB", "#bbe1bb")
box(5.7, 3.0, 2.4, 1.4, "Optional:\n+ n=4 register\ntokens (+3K params)", "#fde2a7")
box(8.4, 3.0, 2.4, 1.4, "PT INT8 quantize\n(no calibration\nno architectural fix)", "#f5b7b1")
box(11.1, 3.0, 2.4, 1.4, "AST INT8\n86 MB ($-74\\%$)\ncos $\\geq 0.99$", "#aed6f1")

arrows_top = [(2.7, 3.7, 3.0, 3.7), (5.4, 3.7, 5.7, 3.7),
              (8.1, 3.7, 8.4, 3.7), (10.8, 3.7, 11.1, 3.7)]
for x1, y1, x2, y2 in arrows_top:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle="->", mutation_scale=14, color="black"))

# Stage labels
ax.text(1.5, 4.7, "(a) Pretraining", ha="center", fontsize=10, fontweight="bold")
ax.text(4.2, 4.7, "(b) Fine-tune\n(stability)", ha="center", fontsize=10, fontweight="bold")
ax.text(6.9, 4.7, "(c) Optional\nstabiliser", ha="center", fontsize=10, fontweight="bold")
ax.text(9.6, 4.7, "(d) Quantize\n(once)", ha="center", fontsize=10, fontweight="bold")
ax.text(12.3, 4.7, "(e) Ship", ha="center", fontsize=10, fontweight="bold")

# === Deployment tier boxes (below) ===
# Show that the SAME INT8 model fans out to multiple tiers
ax.add_patch(FancyArrowPatch((12.3, 3.0), (12.3, 1.7),
                              arrowstyle="->", mutation_scale=18,
                              color="#1565C0", linewidth=2))

# Deployment tier band
deploy_y = 0.4
deploy_h = 1.2
tiers = [
    (0.3, "Hearing aids\n128-256 MB",                 "#9CCC65"),
    (3.0, "IoT sensors,\nwearables\n256-512 MB",      "#9CCC65"),
    (5.7, "Mid-range\nsmartphones\n4 GB",              "#FFEE58"),
    (8.4, "Smart speakers,\nedge appliances\n1-2 GB",  "#FFEE58"),
    (11.1, "Cloud GPU\nmulti-tenant\n40 GB / A100",     "#26C6DA"),
]
for x, label, color in tiers:
    box(x, deploy_y, 2.4, deploy_h, label, color, textsize=8)

# tier labels
ax.text(0.3, deploy_y + deploy_h + 0.15,
        "tier admitted by INT8 size reduction (74%)",
        fontsize=8, color="#1565C0", style="italic")

# bottom-left: what each tier gets
ax.text(7, -0.05,
        "Same INT8 checkpoint runs in all tiers; each device picks the backend "
        "(CoreML / ONNX-RT ARM / TFLite / TensorRT / fbgemm).",
        ha="center", fontsize=8, color="#555")

ax.set_title("Fig.\\,1.  AST INT8 deployment pipeline.  A single quantized checkpoint admits AST-class capability into edge tiers (left, green) "
             "previously confined to CNN-scale models, while simultaneously enabling 4$\\times$ multi-tenant density on cloud GPUs (right).",
             fontsize=10, pad=10, wrap=True)

plt.tight_layout()
out = ASSETS / "fig_deployment_pipeline.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
