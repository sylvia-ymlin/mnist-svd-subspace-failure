import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style
from mnist_logic import load_mnist_data, compute_digit_bases, get_projection_residual

apply_global_style()

def plot_singular_values(train_data, labels):
    """Visualize the magnitude decay of singular values for each digit class."""
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    
    for digit in range(10):
        # Subset data and compute SVD directly for singular value analysis
        digit_data = train_data[:, labels == digit][:, :400]
        _, S, _ = np.linalg.svd(digit_data, full_matrices=False)
        
        # Highlight extreme cases (simple vs complex structure)
        if digit == 1:
            ax.semilogy(range(1, 51), S[:50], color="#D08770", linewidth=2.5, label="Digit 1 (simple)")
        elif digit == 8:
            ax.semilogy(range(1, 51), S[:50], color="#5E81AC", linewidth=2.5, label="Digit 8 (complex)")
        else:
            ax.semilogy(range(1, 51), S[:50], color="#cccccc", alpha=0.5)

    ax.set_xlabel("Singular value index j")
    ax.set_ylabel("Singular value magnitude")
    ax.set_title("SVD Singular Value Decay per Digit Class")
    ax.legend()
    plt.savefig("figures/singular_value_decay.png")
    plt.close()

def plot_basis_grid(bases):
    """Visualize the first 3 basis vectors (singular images) for all digits."""
    fig, axes = plt.subplots(3, 10, figsize=STYLE["figsize_wide"])
    for d in range(10):
        U = bases[d]
        for i in range(3):
            # Reshape 784-dim basis vector back to 28x28
            img = U[:, i].reshape(28, 28).T
            axes[i, d].imshow(img, cmap='gray')
            axes[i, d].axis('off')
            if i == 0:
                axes[i, d].set_title(f"D {d}")

    plt.tight_layout()
    plt.savefig("figures/basis_components_grid.png")
    plt.close()

def plot_reconstructions(test_data, test_labels, bases):
    """Compare original images to rank-22 reconstructions."""
    k = 22
    fig, axes = plt.subplots(2, 10, figsize=STYLE["figsize_wide_short"])

    for d in range(10):
        # Pick the first test example for each class
        idx = np.where(test_labels == d)[0][0]
        original = test_data[:, idx]
        
        # Calculate reconstruction in rank-k subspace
        Uk = bases[d][:, :k]
        recon = Uk @ (Uk.T @ original)

        axes[0, d].imshow(original.reshape(28, 28).T, cmap="gray")
        axes[1, d].imshow(recon.reshape(28, 28).T, cmap="gray")
        axes[0, d].axis("off")
        axes[1, d].axis("off")

    plt.tight_layout()
    plt.savefig("figures/reconstruction_comparison.png")
    plt.close()

def main():
    # --- Step 1: Load Data & Precompute Bases ---
    train_digits, train_labels, test_digits, test_labels = load_mnist_data()
    k_eval = 22
    bases = compute_digit_bases(train_digits, train_labels, rank_limit=k_eval)
    # Actually need rank 50 for singular value plot
    full_bases = compute_digit_bases(train_digits, train_labels, rank_limit=50)

    # --- Step 2: Visualizations ---
    print("Generating visualizations...")
    plot_singular_values(train_digits, train_labels)
    plot_basis_grid(full_bases)
    plot_reconstructions(test_digits, test_labels, full_bases)
    print("Done. Figures saved to figures/.")

if __name__ == "__main__":
    main()
