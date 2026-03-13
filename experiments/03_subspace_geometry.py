"""
Experiment 03: Subspace geometry predicts confusion.

For each pair of digit classes (i, j), compute the minimum principal angle
between their rank-k SVD subspaces. Compare against the confusion matrix.

Run from repo root:
    python experiments/03_subspace_geometry.py

Output:
    figures/principal_angles_heatmap.png
    figures/confusion_heatmap.png
    figures/angles_vs_errors.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.plotting import set_plot_theme, save_figure, NBody_Palette

K = 15
DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

set_plot_theme()

# ── Load data ─────────────────────────────────────────────────────────────────
train_ds, test_ds = get_filtered_mnist(digits=None, data_dir=str(DATA_DIR))
X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_ds, test_ds)

classes = np.arange(10)

# ── Fit rank-k subspaces ──────────────────────────────────────────────────────
print(f"Computing rank-{K} subspaces...")
U_k = {}
for c in classes:
    X_c = X_train[y_train == c]
    U, _, _ = np.linalg.svd(X_c.T, full_matrices=False)
    U_k[c] = U[:, :K]

# ── Compute confusion matrix ──────────────────────────────────────────────────
print("Classifying test set...")
scores = np.zeros((len(X_test), 10))
for c in classes:
    scores[:, c] = np.sum((X_test @ U_k[c]) ** 2, axis=1)
preds = scores.argmax(axis=1)

cm = confusion_matrix(y_test, preds, labels=list(classes))
# Zero diagonal (correct predictions not relevant for confusion analysis)
errors = cm.copy()
np.fill_diagonal(errors, 0)

overall_acc = (preds == y_test).mean()
print(f"Overall accuracy at k={K}: {overall_acc:.4f}")

# ── Compute minimum principal angles between all class pairs ──────────────────
print("Computing pairwise principal angles...")
min_angle_deg = np.zeros((10, 10))
for i in classes:
    for j in classes:
        if i == j:
            min_angle_deg[i, j] = 0.0
        else:
            angles_rad = subspace_angles(U_k[i], U_k[j])
            min_angle_deg[i, j] = np.degrees(angles_rad.min())

# ── Correlation: principal angle vs energy leakage ───────────────────────────
# Small principal angle ↔ high cross-projection energy (mathematical consequence
# of subspace alignment). We verify this empirically to close the chain:
# angle → energy leakage → confusion.
energy_matrix = np.zeros((10, 10))
for i in classes:
    X_test_i = X_test[y_test == i]
    for j in classes:
        proj = X_test_i @ U_k[j]
        energy_matrix[i, j] = np.mean(np.sum(proj**2, axis=1))

off_diag = ~np.eye(10, dtype=bool)
# Energy is theoretically proportional to cos²(θ_min), not θ_min linearly.
# Use cos²(θ) for the correlation to match the underlying geometry.
cos2_angle = np.cos(np.radians(min_angle_deg)) ** 2
r_ae, p_ae = pearsonr(cos2_angle[off_diag], energy_matrix[off_diag])
print(f"cos²(angle) vs energy leakage: r = {r_ae:.4f} (p={p_ae:.2e})")

# ── Figure 1: Principal angles heatmap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(
    min_angle_deg, annot=True, fmt=".1f", ax=ax,
    cmap="Blues", linewidths=0.3,
    xticklabels=classes, yticklabels=classes,
    cbar_kws={"label": "Min. Principal Angle (°)"}
)
ax.set_title(f"Minimum Principal Angle Between Class Subspaces (k={K})")
ax.set_xlabel("Digit Class")
ax.set_ylabel("Digit Class")
fig.tight_layout()
save_figure(fig, "principal_angles_heatmap.png", [FIGURES_DIR])
print("Saved: figures/principal_angles_heatmap.png")
plt.close(fig)

# ── Figure 2: Confusion heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(
    errors, annot=True, fmt="d", ax=ax,
    cmap="Oranges", linewidths=0.3,
    xticklabels=classes, yticklabels=classes,
    cbar_kws={"label": "Misclassification Count"}
)
ax.set_title(f"Confusion Matrix — Off-Diagonal Errors (k={K})")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.tight_layout()
save_figure(fig, "confusion_heatmap.png", [FIGURES_DIR])
print("Saved: figures/confusion_heatmap.png")
plt.close(fig)

# ── Figure 3: Scatter — min angle vs errors ───────────────────────────────────
# One point per ordered pair (i→j), i ≠ j → 90 points total
# Each symmetric angle θ_min(i,j) = θ_min(j,i) appears twice with
# potentially different error counts, revealing directional asymmetry.
angles_flat = []
errors_flat = []
labels_flat = []
for i in classes:
    for j in classes:
        if i != j:
            angles_flat.append(min_angle_deg[i, j])
            errors_flat.append(errors[i, j])
            labels_flat.append(f"{i}→{j}")

angles_flat = np.array(angles_flat)
errors_flat = np.array(errors_flat)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(angles_flat, errors_flat, alpha=0.5, s=30,
           color=NBody_Palette["blue_deep"], edgecolors="none")

# Label the most confused pairs
top_idx = np.argsort(errors_flat)[-8:]
for idx in top_idx:
    if errors_flat[idx] > 0:
        ax.annotate(labels_flat[idx],
                    xy=(angles_flat[idx], errors_flat[idx]),
                    xytext=(2, 3), textcoords="offset points",
                    fontsize=7.5, color=NBody_Palette["gray_dark"])

ax.set_xlabel("Minimum Principal Angle (°)")
ax.set_ylabel("Misclassification Count (i→j)")
ax.set_title(f"Subspace Overlap Predicts Confusion (k={K})")
fig.tight_layout()
save_figure(fig, "angles_vs_errors.png", [FIGURES_DIR])
print("Saved: figures/angles_vs_errors.png")
plt.close(fig)

# ── Print top 10 most confused pairs ─────────────────────────────────────────
print("\nTop 10 confused pairs (i→j):")
top10 = np.argsort(errors_flat)[::-1][:10]
for idx in top10:
    print(f"  {labels_flat[idx]:5s}  errors={errors_flat[idx]:3.0f}  "
          f"angle={angles_flat[idx]:.1f}°")

# ── Asymmetry analysis for the most confused pair (7 → 9) ────────────────────
# θ_min(7,9) = θ_min(9,7) is symmetric, but 7→9 errors exceed 9→7 errors.
# Test: does class 7 have higher variance along the shared principal direction?
i_pair, j_pair = 7, 9
M = U_k[i_pair].T @ U_k[j_pair]           # k×k cross-subspace matrix
U_cross, _, Vt_cross = np.linalg.svd(M)    # S[0] = cos(θ_min)

pv_i = U_k[i_pair] @ U_cross[:, 0]         # principal direction in class 7 subspace
pv_j = U_k[j_pair] @ Vt_cross[0]           # principal direction in class 9 subspace

proj_i = X_train[y_train == i_pair] @ pv_i
proj_j = X_train[y_train == j_pair] @ pv_j
var_i, var_j = np.var(proj_i), np.var(proj_j)

print(f"\nAsymmetry analysis ({i_pair}→{j_pair} vs {j_pair}→{i_pair}):")
print(f"  Var(class {i_pair} along principal dir): {var_i:.4f}")
print(f"  Var(class {j_pair} along principal dir): {var_j:.4f}")
print(f"  Variance ratio {i_pair}/{j_pair}: {var_i/var_j:.3f}")
print(f"  Errors {i_pair}→{j_pair}: {errors[i_pair, j_pair]:.0f}  |  "
      f"Errors {j_pair}→{i_pair}: {errors[j_pair, i_pair]:.0f}")
