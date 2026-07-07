"""Generate the process-flow and architecture diagrams for the PPT (PNG)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "docs"

INK = "#1f2937"
BLUE = "#1d4ed8"
GREEN = "#15803d"
AMBER = "#b45309"
RED = "#b91c1c"
GREY = "#6b7280"


def box(ax, x, y, w, h, text, fc="#eff6ff", ec=BLUE, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color=color))


# ---------------------------------------------------------- 1. PROCESS FLOW
fig, ax = plt.subplots(figsize=(13, 5.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")

box(ax, 0.2, 3.9, 2.5, 1.3, "DATA PILLARS\n1. Borrower behavior\n2. Internal bank DB\n3. Public-domain data",
    fc="#f0fdf4", ec=GREEN, fs=10, bold=True)
box(ax, 3.3, 3.9, 2.3, 1.3, "Feature Engine\nutilization, deltas,\npayment coverage,\nsegments", fs=10)
box(ax, 6.2, 4.35, 3.0, 0.85, "LAYER 1 - Risk Score\nLightGBM (AmEx-winning recipe)", fs=10)
box(ax, 6.2, 3.35, 3.0, 0.85, "LAYER 2 - WHEN model\nhazard -> 12-month PD curve", fs=10)
box(ax, 9.8, 3.9, 3.0, 1.3, "LAYER 3 - RAG Watchlist\nML score + RBI-style\nEWS rule triggers",
    fc="#fff7ed", ec=AMBER, fs=10, bold=True)

box(ax, 1.1, 1.1, 3.0, 1.4, "Reason codes\n(SHAP -> plain language)\n'utilization 95% & rising'", fs=10)
box(ax, 5.0, 1.1, 3.2, 1.4, "CREDIT OFFICER\ndecides & acts\n(human-in-the-loop)",
    fc="#fef2f2", ec=RED, fs=11, bold=True)
box(ax, 9.0, 1.1, 3.4, 1.4, "Actions\nRM call - limit review -\nrestructure - monitor monthly", fs=10)

arrow(ax, 2.7, 4.55, 3.3, 4.55)
arrow(ax, 5.6, 4.55, 6.2, 4.75)
arrow(ax, 5.6, 4.55, 6.2, 3.8)
arrow(ax, 9.2, 4.75, 9.8, 4.6)
arrow(ax, 9.2, 3.78, 9.8, 4.3)
arrow(ax, 10.9, 3.9, 6.8, 2.5)
arrow(ax, 2.6, 2.5, 2.6, 2.5)  # anchor no-op
arrow(ax, 10.6, 3.9, 2.9, 2.5)
arrow(ax, 4.1, 1.8, 5.0, 1.8)
arrow(ax, 8.2, 1.8, 9.0, 1.8)

ax.text(6.5, 5.45, "CreditPulse - monthly re-scoring of every running loan account",
        ha="center", fontsize=13, fontweight="bold", color=INK)
fig.savefig(OUT / "diagram_flow.png", dpi=160, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ---------------------------------------------------------- 2. ARCHITECTURE
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

# sources
box(ax, 0.2, 4.6, 2.4, 1.1, "Core Banking (CBS)\nloan & txn data", fs=10)
box(ax, 0.2, 3.3, 2.4, 1.1, "Credit bureau\n(CIBIL) feeds", fs=10)
box(ax, 0.2, 2.0, 2.4, 1.1, "Public domain\nRBI sectoral, macro", fs=10)

box(ax, 3.2, 2.6, 2.2, 2.4, "Data Lake\nAmazon S3\n+ Glue ETL\n(monthly batch)", fc="#f5f3ff", ec="#6d28d9", fs=10, bold=True)
box(ax, 6.0, 3.6, 2.6, 1.6, "Scoring service\nSageMaker endpoint\nLayer 1 + Layer 2\n(LightGBM)", fc="#f5f3ff", ec="#6d28d9", fs=10, bold=True)
box(ax, 6.0, 1.6, 2.6, 1.4, "Rules engine\nRBI EWS triggers\n(Lambda)", fc="#f5f3ff", ec="#6d28d9", fs=10)
box(ax, 9.3, 3.6, 3.3, 1.6, "CreditPulse dashboard\n+ REST API (API Gateway)\nRAG watchlist, PD curves,\nreason codes", fc="#fff7ed", ec=AMBER, fs=10, bold=True)
box(ax, 9.3, 1.6, 3.3, 1.4, "Case management /\nofficer workflow\n(human decision)", fc="#fef2f2", ec=RED, fs=10, bold=True)

box(ax, 3.2, 0.3, 5.4, 0.9, "Model governance: monthly stability monitoring - per-segment AUC -\nSHAP audit trail - champion/challenger retraining", fc="#f8fafc", ec=GREY, fs=9)

arrow(ax, 2.6, 5.15, 3.3, 4.6)
arrow(ax, 2.6, 3.85, 3.2, 3.85)
arrow(ax, 2.6, 2.55, 3.3, 3.0)
arrow(ax, 5.4, 4.2, 6.0, 4.4)
arrow(ax, 5.4, 3.4, 6.0, 2.4)
arrow(ax, 8.6, 4.4, 9.3, 4.4)
arrow(ax, 8.6, 2.3, 9.3, 2.3)
arrow(ax, 10.9, 3.6, 10.9, 3.0)

ax.text(6.5, 5.95, "Production architecture - AWS sandbox-ready, fully open-source, on-prem deployable",
        ha="center", fontsize=13, fontweight="bold", color=INK)
ax.text(6.5, 5.55, "Prototype today: same code deployed on Streamlit Cloud from the public GitHub repo",
        ha="center", fontsize=10, color=GREY)
fig.savefig(OUT / "diagram_arch.png", dpi=160, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("Saved diagrams:", OUT / "diagram_flow.png", OUT / "diagram_arch.png")