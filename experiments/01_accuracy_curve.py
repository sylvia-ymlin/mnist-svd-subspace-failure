"""
Experiment 1: Accuracy vs rank k + saturation rank per digit.

Run from repo root:
    python experiments/01_accuracy_vs_k.py

Output: figures/accuracy_vs_k.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.plotting import set_plot_theme, save_figure

# ── Config ────────────────────────────────────────────────────────────────────
K_VALUES             = list(range(5, 51))
SATURATION_THRESHOLD = 0.98
DATA_DIR             = Path(__file__).parent.parent / "data"
FIGURES_DIR          = Path(__file__).parent.parent / "figures"

set_plot_theme()

# ── Load data ─────────────────────────────────────────────────────────────────
train_ds, test_ds = get_filtered_mnist(digits=None, data_dir=str(DATA_DIR))
X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_ds, test_ds)

# ── Full SVD per class ────────────────────────────────────────────────────────
classes = np.arange(10)
print("Computing full SVD for each digit class...")
U_full = {}
for c in classes:
    X_c = X_train[y_train == c]
    U, _, _ = np.linalg.svd(X_c.T, full_matrices=False)
    U_full[c] = U
    print(f"  Digit {c}: {X_c.shape[0]} samples, U shape {U.shape}")

# ── Evaluate accuracy for each k ──────────────────────────────────────────────
n_per_class    = {c: np.sum(y_test == c) for c in classes}
correct_counts = {c: np.zeros(len(K_VALUES)) for c in classes}

print(f"\nEvaluating over k = {K_VALUES[0]}..{K_VALUES[-1]}...")
for k_idx, k in enumerate(K_VALUES):
    scores = np.zeros((len(X_test), 10))
    for c in classes:
        U_k_c      = U_full[c][:, :k]
        scores[:, c] = np.sum((X_test @ U_k_c) ** 2, axis=1)
    preds = scores.argmax(axis=1)
    for c in classes:
        mask = y_test == c
        correct_counts[c][k_idx] = (preds[mask] == c).sum()
    if (k_idx + 1) % 10 == 0 or k == K_VALUES[-1]:
        print(f"  k={k:3d}  overall accuracy = {(preds == y_test).mean():.4f}")

correct_rates = {c: correct_counts[c] / n_per_class[c] for c in classes}

# ── Saturation rank k* per digit ──────────────────────────────────────────────
saturation = {}
for c in classes:
    rates = correct_rates[c]
    k_star_idx = np.argmax(rates >= SATURATION_THRESHOLD * rates.max())
    saturation[c] = {
        "k_star":      K_VALUES[k_star_idx],
        "acc_at_kstar": rates[k_star_idx],
    }
    print(f"  Digit {c}: k* = {saturation[c]['k_star']:2d}, "
          f"acc = {saturation[c]['acc_at_kstar']:.3f}")

# ── Figure: Accuracy heatmap with k* overlay ──────────────────────────────────
sorted_by_acc = sorted(classes, key=lambda c: correct_rates[c].mean())
acc_matrix    = np.array([correct_rates[c] for c in sorted_by_acc])

fig, ax = plt.subplots(figsize=(11, 4.5))

sns.heatmap(
    acc_matrix, ax=ax, cmap="YlGnBu",
    vmin=acc_matrix.min(), vmax=1.0,
    xticklabels=[k if k % 5 == 0 else "" for k in K_VALUES],
    yticklabels=[f"Digit {c}" for c in sorted_by_acc],
    linewidths=0.3, linecolor="white",
    cbar_kws={"label": "Accuracy", "shrink": 0.85},
)

# Determine the global operating point k_op = max(k*) over all digits
max_kstar = max(s["k_star"] for s in saturation.values())
k_op_idx  = K_VALUES.index(max_kstar)

# Overlay operating point k_op
ax.axvline(k_op_idx, color="tomato", linewidth=1.5, linestyle="--", alpha=0.8)

# Overlay saturation rank k* as star markers
for i, c in enumerate(sorted_by_acc):
    k_star     = saturation[c]["k_star"]
    k_star_idx = K_VALUES.index(k_star)
    # the heatmap cells are centered at integer + 0.5
    ax.scatter(k_star_idx + 0.5, i + 0.5, marker="*", color="tomato", s=100,
               edgecolor="white", linewidth=0.5, zorder=10)

# Custom legend for the stars and the line
legend_elements = [
    mlines.Line2D([0], [0], marker="*", color="w", markerfacecolor="tomato", markersize=12,
                  markeredgecolor="w", label=f"Saturation Rank $k^*$ per digit ($98\%$ max)"),
    mlines.Line2D([0], [0], color="tomato", linestyle="--", linewidth=1.5,
                  label=f"Max $k^* = {max_kstar}$ (Operating point)")
]
ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10, frameon=False)

ax.set_xlabel("Rank $k$")
ax.set_ylabel("")
ax.set_title("Accuracy vs Rank $k$ (per digit)")
ax.tick_params(axis="y", labelsize=9)

fig.tight_layout()
save_figure(fig, "accuracy_vs_k.png", [FIGURES_DIR])
print("\nSaved: figures/accuracy_vs_k.png")
plt.close(fig)
