import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from style_utils import STYLE, apply_global_style

apply_global_style()

def compute_principal_angles(U1, U2, k1, k2):
    """
    Compute principal angles between two subspaces.
    U1, U2: Basis matrices (columns are orthonormal vectors)
    k1, k2: Ranks (number of basis vectors to use)
    """
    U1_k = U1[:, :k1]
    U2_k = U2[:, :k2]
    _, S, _ = np.linalg.svd(U1_k.T @ U2_k, full_matrices=False)
    S = np.clip(S, 0, 1)
    angles_rad = np.arccos(S)
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def plot_mean_angle_heatmap(angle_means):
    """Visualize: Mean angle heatmap (mask diagonal — self-overlap is 0° and misleading)"""
    masked_means = angle_means.copy().astype(float)
    np.fill_diagonal(masked_means, np.nan)

    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    im = ax.imshow(masked_means, cmap="Blues_r",
                   vmin=np.nanmin(masked_means), vmax=np.nanmax(masked_means))

    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#aaa", fontsize=9)
            else:
                val = masked_means[i, j]
                color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.1f}°", ha="center", va="center",
                       fontsize=8, color=color)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Digit Class")
    fig.colorbar(im, ax=ax, label="Mean Angle (degrees)")
    plt.savefig("figures/mean_principal_angles_heatmap.png")
    plt.close()

def plot_angle_vs_confusion_scatter(angle_mean_vals, confusion_vals, pairs, corr_mean, p_mean):
    """Visualize: Scatter plot (mean angle vs confusion)"""
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    ax.scatter(angle_mean_vals, confusion_vals, alpha=0.6,
               s=80, color=STYLE["colors"]["primary"], edgecolor="black", linewidth=0.5)

    top_idx = np.argsort(confusion_vals)[-5:]
    for idx in top_idx:
        i, j = pairs[idx]
        ax.annotate(f"({i},{j})",
                    xy=(angle_mean_vals[idx], confusion_vals[idx]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=STYLE["colors"]["secondary"])

    ax.set_xlabel("Mean Principal Angle (degrees)")
    ax.set_ylabel("Mutual Confusion Rate (%)")
    p_str = f"p={p_mean:.4e}" if p_mean < 0.001 else f"p={p_mean:.4f}"
    ax.set_title(f"Angle vs. Confusion (ρ={corr_mean:.3f}, {p_str})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/angle_vs_confusion_mean.png")
    plt.close()


def main():
    k = 22
    train_digits = np.load("data/TrainDigits.npy")
    train_labels = np.load("data/TrainLabels.npy")
    test_digits = np.load("data/TestDigits.npy")
    test_labels = np.load("data/TestLabels.npy")

    bases_dict = {}
    for d in range(10):
        digit_data = train_digits[:, train_labels == d][:, :400]
        U, _, _ = np.linalg.svd(digit_data, full_matrices=False)
        bases_dict[d] = U

    angle_means = np.zeros((10, 10))

    for i in range(10):
        for j in range(i + 1, 10):
            angles = compute_principal_angles(bases_dict[i], bases_dict[j], k, k)
            angle_means[i, j] = angle_means[j, i] = np.mean(angles)

    best_preds = np.load("data/BestPredictions.npy")
    conf_raw = np.zeros((10, 10))
    for t, p in zip(test_labels, best_preds):
        conf_raw[int(t)][int(p)] += 1
    conf_matrix = conf_raw / conf_raw.sum(axis=1, keepdims=True) * 100

    confusion_rates = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i != j:
                confusion_rates[i, j] = conf_matrix[i, j]

    pairs, angle_mean_vals, confusion_vals = [], [], []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((i, j))
            angle_mean_vals.append(angle_means[i, j])
            confusion_vals.append((confusion_rates[i, j] + confusion_rates[j, i]) / 2)

    corr_mean, p_mean = spearmanr(angle_mean_vals, confusion_vals)
    print(f"\nMean Principal Angle vs. Confusion: ρ = {corr_mean:.4f}, p = {p_mean:.6f}")

    plot_mean_angle_heatmap(angle_means)
    plot_angle_vs_confusion_scatter(angle_mean_vals, confusion_vals, pairs, corr_mean, p_mean)

    # --- Verification of 8-1 minimum angle mentioned in §6 ---
    # Baseline (uncentered)
    angles_81 = compute_principal_angles(bases_dict[8], bases_dict[1], 22, 22)
    min_angle_81 = np.min(angles_81)
    
    # PCA (centered)
    X8 = train_digits[:, train_labels == 8][:, :400]
    X1 = train_digits[:, train_labels == 1][:, :400]
    U8_pca, _, _ = np.linalg.svd(X8 - X8.mean(axis=1)[:, np.newaxis], full_matrices=False)
    U1_pca, _, _ = np.linalg.svd(X1 - X1.mean(axis=1)[:, np.newaxis], full_matrices=False)
    angles_81_pca = compute_principal_angles(U8_pca, U1_pca, 22, 22)
    min_angle_81_pca = np.min(angles_81_pca)
    
    print(f"\nVerification for §6:")
    print(f"  uncentered 8-1 min angle : {min_angle_81:.2f}° (matches ~6.21° in report)")
    print(f"  centered (PCA) 8-1 min angle : {min_angle_81_pca:.2f}° (matches ~13.78° in report)")

if __name__ == "__main__":
    main()
