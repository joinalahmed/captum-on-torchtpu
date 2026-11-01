"""Render the LinkedIn card for the Captum-on-TorchTPU post into images/linkedin_card.png.

Numbers come from FINDINGS.md. Run from the project root:
    .venv/bin/python scripts/make_linkedin_image.py
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

OUT = pathlib.Path(__file__).resolve().parents[1] / "images"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(12, 12), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis("off")

ax.text(0.8, 11.0, "CAPTUM ON CLOUD TPUs WITH TORCHTPU", fontsize=15, color=BLUE, fontweight="bold",
        fontfamily=FONT, va="center")
ax.text(0.8, 10.1, "A few days of attribution on TPUs,\nwritten up as a work log", fontsize=34,
        color=INK, fontweight="bold", fontfamily=FONT, va="center", linespacing=1.15)

tiles = [
    ("38", "captum.attr methods run on\ntorch.device(\"tpu\") unmodified;\n27 match CPU to ~1e-7", BLUE),
    ("1e-3 → 1e-7", "attribution error vs CPU after\ntorch.set_float32_matmul_precision\n(\"highest\"), at no cost", AQUA),
    ("0.80 / 0.36 / 0.16 s", "Layer IG on Qwen3-0.6B, 64 steps:\nstrict eager / DeferAndFuse /\ntorch.compile. CPU: 4 to 17 s", ORANGE),
    ("5 PRs", "open on torch_tpu and Captum,\nfive more branches tracked by\nissues; all validated on a v6e-4", BLUE),
]
x0, y0, w, h, gap = 0.8, 4.3, 5.0, 2.25, 0.4
for i, (big, small, col) in enumerate(tiles):
  x = x0 + (i % 2) * (w + gap); y = y0 + (1 - i // 2) * (h + gap)
  ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                              facecolor="white", edgecolor=GRID, linewidth=1.2))
  ax.plot([x + 0.35, x + 0.35], [y + 0.45, y + h - 0.45], color=col, linewidth=5, solid_capstyle="round")
  ax.text(x + 0.65, y + h - 0.55, big, fontsize=26 if len(big) < 10 else 21, color=INK, fontweight="bold",
          fontfamily=FONT, va="top")
  ax.text(x + 0.65, y + h - 1.25, small, fontsize=12.5, color=INK2, fontfamily=FONT, va="top", linespacing=1.3)

ax.text(0.8, 3.35, "Eager attribution is launch-bound: one call is 363 XLA programs and a 2% TensorCore duty cycle.",
        fontsize=14, color=INK2, fontfamily=FONT, va="center")
ax.text(0.8, 2.75, "Found and fixed on the way: a scalar buffer leak, a missing max_unpool2d, a compiled-mode\n"
        "view bug, a Captum device bug, eight unregistered ops, 11 stale deny-list entries.",
        fontsize=14, color=INK2, fontfamily=FONT, va="top", linespacing=1.35)

ax.plot([0.8, 11.2], [1.45, 1.45], color=GRID, linewidth=1)
ax.text(0.8, 0.95, "github.com/joinalahmed/captum-on-torchtpu", fontsize=14, color=BLUE, fontfamily=FONT,
        fontweight="bold", va="center")
ax.text(11.2, 0.95, "Joinal Ahmed", fontsize=14, color=MUTED, fontfamily=FONT, va="center", ha="right")

fig.savefig(OUT / "linkedin_card.png", dpi=100, facecolor=SURFACE)
print("wrote", OUT / "linkedin_card.png")
