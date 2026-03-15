"""
Experiment 02: Energy leakage and confusion correlation.
Computes inter-class projection energy and correlates it with classifier confusion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.plotting import set_plot_theme, save_figure

K = 15
DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

set_plot_theme()

def main():
    # 1. Load data
    train_ds, test_ds = get_filtered_mnist(digits=None, data_dir=str(DATA_DIR))
    X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_ds, test_ds)
    classes = np.arange(10)

    # 2. Compute SVD Subspaces
    print(f"Computing rank-{K} SVD subspaces...")
    U_k = {}
    for c in classes:
        Xc = X_train[y_train == c]
        U, _, _ = np.linalg.svd(Xc.T, full_matrices=False)
        U_k[c] = U[:, :K]

    # 3. Compute Energy and Confusion Matrices
    print("Computing Energy and Confusion Matrices...")
    energy_matrix = np.zeros((10, 10))
    confusion_matrix = np.zeros((10, 10))

    for i in classes:
        X_test_i = X_test[y_test == i]
        for j in classes:
            proj = X_test_i @ U_k[j]
            energy_matrix[i, j] = np.mean(np.sum(proj**2, axis=1))

    # Predictions for confusion matrix
    scores = np.zeros((len(X_test), 10))
    for c in classes:
        scores[:, c] = np.sum((X_test @ U_k[c]) ** 2, axis=1)
    preds = scores.argmax(axis=1)
    
    for i in classes:
        for j in classes:
            confusion_matrix[i, j] = np.sum((preds == j) & (y_test == i))

    # 4. Figure: Energy Matrix Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax_off = np.max(energy_matrix[~np.eye(10, dtype=bool)])
    sns.heatmap(energy_matrix, annot=True, fmt=".1f", ax=ax, cmap="Purples", 
                vmin=0, vmax=vmax_off*1.5, annot_kws={"size": 9})
    ax.set_title(f"Inter-class Projection Energy (k={K})")
    ax.set_xlabel("Subspace Class")
    ax.set_ylabel("True Image Class")
    plt.tight_layout()
    save_figure(fig, "projection_energy_matrix.png", [FIGURES_DIR])

    # 5. Figure: Correlation Scatter Plot
    # Normalize confusion counts to per-class error rates so the metric is
    # comparable to the per-image-mean energy matrix.  Raw counts inflate the
    # correlation for large classes even when their per-sample leakage is similar.
    off_diag = ~np.eye(10, dtype=bool)
    n_per_class = np.array([np.sum(y_test == c) for c in classes])
    error_rate_matrix = confusion_matrix / n_per_class[:, None]   # row-normalize

    energy_flat = energy_matrix[off_diag]
    confusion_flat = error_rate_matrix[off_diag]
    r, p_val = pearsonr(energy_flat, confusion_flat)
    print(f"Pearson correlation r: {r:.4f} (p={p_val:.2e})")

    plt.figure(figsize=(7, 6))
    plt.scatter(energy_flat, confusion_flat, alpha=0.7, color='indigo')
    z = np.polyfit(energy_flat, confusion_flat, 1)
    p = np.poly1d(z)
    plt.plot(energy_flat, p(energy_flat), "r--", alpha=0.8, label=f'Trendline (r={r:.2f})')

    # Label top 3 confused
    top_indices = np.argsort(confusion_flat)[-3:]
    row_idx, col_idx = np.where(off_diag)
    for idx in top_indices:
        plt.annotate(f'{row_idx[idx]}\u2192{col_idx[idx]}', (energy_flat[idx], confusion_flat[idx]),
                     xytext=(5,5), textcoords="offset points")

    plt.xlabel('Inter-class Projection Energy')
    plt.ylabel('Misclassification Rate (per class)')
    plt.title('Energy leakage vs. Prediction Confusion')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure(plt.gcf(), "energy_vs_confusion.png", [FIGURES_DIR])

if __name__ == "__main__":
    main()
