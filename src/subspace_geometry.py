import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from style_utils import STYLE, apply_global_style
from mnist_logic import load_mnist_data, compute_digit_bases, get_projection_residual, get_confusion_rates

apply_global_style()

def compute_principal_angles(U1, U2, k):
    """Compute principal angles theta = arccos(singular_values(U1_k^T * U2_k))."""
    U1_k = U1[:, :k]
    U2_k = U2[:, :k]
    _, S, _ = np.linalg.svd(U1_k.T @ U2_k, full_matrices=False)
    # Clip for stability then convert to degrees
    S = np.clip(S, 0, 1)
    return np.degrees(np.arccos(S))

def plot_angle_heatmap(angle_matrix):
    """Visualize mean principal angles as a heatmap."""
    display_matrix = angle_matrix.copy().astype(float)
    np.fill_diagonal(display_matrix, np.nan)

    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    im = ax.imshow(display_matrix, cmap="Blues_r",
                   vmin=np.nanmin(display_matrix), vmax=np.nanmax(display_matrix))

    for i in range(10):
        for j in range(10):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#aaa", fontsize=9)
            else:
                val = display_matrix[i, j]
                ax.text(j, i, f"{val:.1f}°", ha="center", va="center",
                       fontsize=8, color="white" if val < 50 else "black")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Digit Class")
    fig.colorbar(im, ax=ax, label="Mean Angle (degrees)")
    plt.savefig("figures/mean_principal_angles_heatmap.png")
    plt.close()

def plot_angle_vs_confusion(angles, confusions, correlation):
    """Visualize correlation between subspace proximity and classification errors."""
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    ax.scatter(angles, confusions, alpha=0.6, s=80, 
               color=STYLE["colors"]["primary"], edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Mean Principal Angle (degrees)")
    ax.set_ylabel("Mutual Confusion Rate (%)")
    ax.set_title(f"Angle vs. Confusion (Spearman rho = {correlation:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/angle_vs_confusion_mean.png")
    plt.close()

def main():
    k_eval = 22
    # --- Step 1: Load Data & Pre-computed predictions ---
    train_digits, train_labels, _, test_labels = load_mnist_data()
    best_preds = np.load("data/BestPredictions.npy")

    # --- Step 2: Compute SVD Bases ---
    bases = compute_digit_bases(train_digits, train_labels, num_samples=400, rank_limit=k_eval)

    # --- Step 3: Compute Pairwise Metric ---
    mean_angles = np.zeros((10, 10))
    for i in range(10):
        for j in range(i + 1, 10):
            angles = compute_principal_angles(bases[i], bases[j], k_eval)
            mean_angles[i, j] = mean_angles[j, i] = np.mean(angles)

    # --- Step 4: Confusion Analysis ---
    conf_matrix = get_confusion_rates(test_labels, best_preds)
    angle_list, confusion_list = [], []
    for i in range(10):
        for j in range(i + 1, 10):
            angle_list.append(mean_angles[i, j])
            # Mutual confusion: averaged error rate between two classes
            confusion_list.append((conf_matrix[i, j] + conf_matrix[j, i]) / 2)

    # --- Step 5: Results & Plotting ---
    corr, _ = spearmanr(angle_list, confusion_list)
    print(f"Geometric Correlation (Angle vs Confusion): rho = {corr:.4f}")
    plot_angle_heatmap(mean_angles)
    plot_angle_vs_confusion(angle_list, confusion_list, corr)

    # --- Step 6: Specific Verification for PCA Section ---
    # Center 8 and 1 to compute PCA angles manually for comparison
    X8 = train_digits[:, train_labels == 8][:, :400]
    X1 = train_digits[:, train_labels == 1][:, :400]
    U8_pca, _, _ = np.linalg.svd(X8 - X8.mean(axis=1)[:, np.newaxis], full_matrices=False)
    U1_pca, _, _ = np.linalg.svd(X1 - X1.mean(axis=1)[:, np.newaxis], full_matrices=False)
    
    min_angle_uncentered = np.min(compute_principal_angles(bases[8], bases[1], 22))
    min_angle_pca = np.min(compute_principal_angles(U8_pca, U1_pca, 22))
    
    print(f"\nVerification (Digit Pair 8-1):")
    print(f"  Uncentered Min Angle: {min_angle_uncentered:.2f} deg")
    print(f"  PCA Centered Min Angle: {min_angle_pca:.2f} deg")

if __name__ == "__main__":
    main()
