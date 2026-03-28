import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style

apply_global_style()

def plot_accuracy_curve(k_values, accuracies, k_best, acc_best):
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    ax.plot(k_values, accuracies, color=STYLE["colors"]["primary"],
            linewidth=2, marker="o", markersize=3, markerfacecolor="white",
            markeredgewidth=1.2, markeredgecolor=STYLE["colors"]["primary"], zorder=2)
    
    # Highlight peak
    ax.plot(k_best, acc_best, "o", markersize=9, color=STYLE["colors"]["secondary"], zorder=3)
    ax.annotate(
        f"{acc_best:.2f}% ($k$={k_best})", 
        xy=(k_best, acc_best),
        xytext=(k_best + 5, acc_best + 0.5), 
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#666", lw=1),
        color=STYLE["colors"]["secondary"], 
        fontweight="bold"
    )
    
    ax.set_xlabel("Subspace rank $k$")
    ax.set_ylabel("Classification accuracy (%)")
    plt.savefig("figures/accuracy_vs_rank.png")
    plt.close()

def plot_confusion_matrix(cm, highlight_top_n=3):
    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > 50 else STYLE["colors"]["text"]
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    # Highlight primary errors: 5->3, 8->1, 7->9, 9->7
    highlight_coords = [(5, 3), (8, 1), (7, 9), (9, 7)]
    for row, col in highlight_coords:
        ax.add_patch(plt.Rectangle(
            (col - 0.5, row - 0.5), 1, 1, 
            linewidth=2, edgecolor=STYLE["colors"]["secondary"],
            facecolor="none", zorder=5
        ))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Rate (%)")
    plt.savefig("figures/confusion_matrix.png")
    plt.close()

def main():
    # Load data
    train_digits = np.load("data/TrainDigits.npy")
    train_labels = np.load("data/TrainLabels.npy")
    test_digits = np.load("data/TestDigits.npy")
    test_labels = np.load("data/TestLabels.npy")
    
    # Pre-compute bases and squared norms for all test digits
    bases = []
    for d in range(10):
        # Using 400 training samples per class to build basis
        digit_data = train_digits[:, train_labels == d][:, :400]
        U, _, _ = np.linalg.svd(digit_data, full_matrices=False)
        bases.append(U[:, :50])
    
    test_norm_sq = np.sum(test_digits**2, axis=0)

    # Compute all projection coefficients once: (10 classes, 50 ranks, 10000 images)
    # This stores the squared dot products: (u_j^T d)^2
    all_coeffs_sq = []
    for B in bases:
        # B is (784, 50), test_digits is (784, 10000)
        # coeffs is (50, 10000)
        coeffs = B.T @ test_digits
        all_coeffs_sq.append(coeffs**2)
    
    # Global k sweep using cumulative sum of squared coefficients
    ks = range(1, 51)
    global_results = []
    
    # Pre-calculate all distance squares for all k: (10 classes, 50 ranks, 10000 images)
    # dist_sq = test_norm_sq - sum(coeffs_sq up to k)
    all_dist_sq = []
    for class_idx in range(10):
        # cum_norm_sq is (50, 10000)
        cum_norm_sq = np.cumsum(all_coeffs_sq[class_idx], axis=0)
        # residuals_sq is (50, 10000)
        residuals_sq = np.maximum(0, test_norm_sq[np.newaxis, :] - cum_norm_sq)
        all_dist_sq.append(residuals_sq)
    
    all_dist_sq = np.array(all_dist_sq) # (10, 50, 10000)

    for k_idx, k in enumerate(ks):
        # For this k, distances to all classes: (10, 10000)
        dists_at_k = all_dist_sq[:, k_idx, :]
        preds = np.argmin(dists_at_k, axis=0)
        global_results.append(np.mean(preds == test_labels) * 100)

    best_k_idx = np.argmax(global_results)
    best_k = ks[best_k_idx]
    best_acc = global_results[best_k_idx]
    
    best_preds = np.argmin(all_dist_sq[:, best_k_idx, :], axis=0)

    conf = np.zeros((10, 10))
    for t, p in zip(test_labels, best_preds):
        conf[t][p] += 1
    conf_norm = (conf / conf.sum(axis=1)[:, np.newaxis]) * 100

    plot_accuracy_curve(ks, global_results, best_k, best_acc)
    plot_confusion_matrix(conf_norm)
    
    # Save best globally-k preds for analysis script
    np.save("data/BestPredictions.npy", best_preds)

if __name__ == "__main__":
    main()
