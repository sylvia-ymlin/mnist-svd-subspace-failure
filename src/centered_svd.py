import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style
from mnist_logic import load_mnist_data, compute_digit_bases, get_projection_residual, get_confusion_rates

apply_global_style()

def run_classification(test_data, test_labels, bases, means, k):
    """Assign digit class based on minimum residual (supports PCA mode if means is provided)."""
    preds = []
    for i in range(test_data.shape[1]):
        image = test_data[:, i]
        # Calculate residuals for all 10 classes, optionally using class means for centering
        res = [get_projection_residual(image, bases[d], k, mean=means[d] if means else None) for d in range(10)]
        preds.append(np.argmin(res))
    
    accuracy = np.mean(np.array(preds) == test_labels) * 100
    conf_norm = get_confusion_rates(test_labels, preds)
    return accuracy, conf_norm

def plot_accuracy_comparison(ks, svd_accs, pca_accs):
    """Visualize accuracy improvements from centering (PCA)."""
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    ax.plot(ks, svd_accs, color=STYLE["colors"]["primary"], label="SVD (uncentered)", marker="o")
    ax.plot(ks, pca_accs, color=STYLE["colors"]["secondary"], label="PCA (centered)", marker="s")
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plt.savefig("figures/pca_accuracy_vs_rank.png")
    plt.close()

def plot_confusion_comparison(conf_svd, conf_pca):
    """Display side-by-side confusion matrices for SVD vs PCA."""
    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_pair"])
    for ax, conf, title in zip(axes, [conf_svd, conf_pca], ["SVD (Baseline)", "PCA (Centered)"]):
        ax.imshow(conf, cmap="Blues", vmin=0, vmax=100)
        ax.set_title(title)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                val = conf[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                        fontsize=8, color="white" if val > 50 else "black")

    plt.tight_layout()
    plt.savefig("figures/pca_confusion_comparison.png")
    plt.close()

def main():
    # --- Step 1: Load Data & Precompute Bases ---
    train_digits, train_labels, test_digits, test_labels = load_mnist_data()
    
    bases_svd = compute_digit_bases(train_digits, train_labels, num_samples=400, rank_limit=50)
    
    # Calculate means and centered bases for PCA
    means, bases_pca = [], []
    for d in range(10):
        X = train_digits[:, train_labels == d][:, :400]
        mu = X.mean(axis=1)
        means.append(mu)
        U_centered, _, _ = np.linalg.svd(X - mu[:, np.newaxis], full_matrices=False)
        bases_pca.append(U_centered)

    # --- Step 2: Performance Comparison Sweep ---
    ks = [5, 10, 20, 22, 23, 30, 40]
    svd_accuracies, pca_accuracies = [], []
    
    print("Comparing SVD vs PCA performance...")
    for k in ks:
        acc_svd, _ = run_classification(test_digits, test_labels, bases_svd, None, k)
        acc_pca, _ = run_classification(test_digits, test_labels, bases_pca, means, k)
        svd_accuracies.append(acc_svd)
        pca_accuracies.append(acc_pca)
        print(f"k={k:2d} | SVD: {acc_svd:.2f}% | PCA: {acc_pca:.2f}%")

    # --- Step 3: Detailed Evaluation at Optimal Ranks ---
    final_acc_svd, final_conf_svd = run_classification(test_digits, test_labels, bases_svd, None, 22)
    final_acc_pca, final_conf_pca = run_classification(test_digits, test_labels, bases_pca, means, 23)
    
    print(f"\nFinal Comparison Result:")
    print(f"  SVD (k=22) accuracy: {final_acc_svd:.2f}%")
    print(f"  PCA (k=23) accuracy: {final_acc_pca:.2f}%")

    # --- Step 4: Visualizations ---
    plot_accuracy_comparison(ks, svd_accuracies, pca_accuracies)
    plot_confusion_comparison(final_conf_svd, final_conf_pca)

if __name__ == "__main__":
    main()
