import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style
from mnist_logic import load_mnist_data, compute_digit_bases, get_projection_residual, get_confusion_rates

apply_global_style()

def classify_images(test_data, bases, k):
    """Assign digit class based on minimum subspace residual."""
    predictions = []
    for i in range(test_data.shape[1]):
        image = test_data[:, i]
        # Direct implementation of r = ||d - Uk * Uk^T * d||_2
        residuals = [get_projection_residual(image, bases[d], k) for d in range(10)]
        predictions.append(np.argmin(residuals))
    return np.array(predictions)

def plot_accuracy_curve(k_values, accuracies):
    """Visualize accuracy across different ranks k."""
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    
    best_idx = np.argmax(accuracies)
    k_best, acc_best = k_values[best_idx], accuracies[best_idx]

    ax.plot(k_values, accuracies, color=STYLE["colors"]["primary"],
            linewidth=2, marker="o", markersize=3, markerfacecolor="white",
            markeredgewidth=1.2, markeredgecolor=STYLE["colors"]["primary"], zorder=2)
    
    # Highlight peak accuracy
    ax.plot(k_best, acc_best, "o", markersize=9, color=STYLE["colors"]["secondary"], zorder=3)
    ax.annotate(f"{acc_best:.2f}% (k={k_best})", xy=(k_best, acc_best),
                xytext=(k_best + 5, acc_best + 0.5), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#666", lw=1),
                color=STYLE["colors"]["secondary"], fontweight="bold")
    
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Classification accuracy (%)")
    plt.savefig("figures/accuracy_vs_rank.png")
    plt.close()

def plot_confusion_matrix(conf_matrix):
    """Visualize normalized confusion matrix at optimal rank."""
    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    im = ax.imshow(conf_matrix, cmap="Blues", vmin=0, vmax=100)

    for i in range(10):
        for j in range(10):
            val = conf_matrix[i, j]
            color = "white" if val > 50 else STYLE["colors"]["text"]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i==j else "normal")

    # Mark digit pairs with notable high confusion as discussed in the report
    failure_pairs = [(5, 3), (8, 1), (7, 9), (9, 7)]
    for row, col in failure_pairs:
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                   linewidth=2, edgecolor=STYLE["colors"]["secondary"],
                                   facecolor="none", zorder=5))

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Rate (%)")
    plt.savefig("figures/confusion_matrix.png")
    plt.close()

def main():
    # --- Step 1: Load Data & Build Bases ---
    train_digits, train_labels, test_digits, test_labels = load_mnist_data()
    # Compute full subspaces up to rank 50 using 400 training samples
    bases = compute_digit_bases(train_digits, train_labels, num_samples=400, rank_limit=50)
    
    # --- Step 2: Rank Selection (k-sweep) ---
    ks = [1, 5, 10, 15, 20, 22, 25, 30, 40, 50]
    accuracies = []
    
    print("Starting accuracy sweep across ranks k...")
    for k in ks:
        preds = classify_images(test_digits, bases, k)
        acc = np.mean(preds == test_labels) * 100
        accuracies.append(acc)
        print(f"  k = {k:2d}: Accuracy = {acc:.2f}%")
    
    # --- Step 3: Final Evaluation at Best k ---
    best_idx = np.argmax(accuracies)
    best_k, best_acc = ks[best_idx], accuracies[best_idx]
    best_preds = classify_images(test_digits, bases, best_k)
    
    conf_norm = get_confusion_rates(test_labels, best_preds)
    plot_accuracy_curve(ks, accuracies)
    plot_confusion_matrix(conf_norm)
    
    # Output result and save predictions for subspace geometry analysis
    np.save("data/BestPredictions.npy", best_preds)
    print(f"\nFinal: Best Accuracy {best_acc:.2f}% at k={best_k}")

if __name__ == "__main__":
    main()
