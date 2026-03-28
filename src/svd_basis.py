import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style

apply_global_style()

def plot_singular_values_comparison(sv_dict):
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])

    highlight = {
        1: ("#D08770", "Digit 1 (simple)"),   # orange
        8: ("#5E81AC", "Digit 8 (complex)"),  # blue
    }

    # Background: all other digits as thin gray lines
    for digit, sv in sorted(sv_dict.items()):
        if digit not in highlight:
            ax.semilogy(range(1, len(sv) + 1), sv,
                        color="#cccccc", linewidth=1.0, alpha=0.8, zorder=1)

    # Foreground: highlighted digits
    for digit, (color, label) in highlight.items():
        sv = sv_dict[digit]
        ax.semilogy(range(1, len(sv) + 1), sv,
                    color=color, linewidth=2.5, label=label, zorder=2)

    # Gray reference line for legend
    ax.plot([], [], color="#cccccc", linewidth=1.5, label="Other digits")

    ax.set_xlim(0, 52)
    ax.set_ylim(0.3, 25)
    ax.set_xlabel("Singular value index $j$")
    ax.set_ylabel("$\\sigma_j$")
    ax.legend(loc="lower left", fontsize=10)
    plt.savefig("figures/singular_value_decay.png")
    plt.close()

def plot_all_singular_images(u_dict):
    """Generates a 3x10 grid of singular images for all digits (3 modes x 10 digits)."""
    fig, axes = plt.subplots(3, 10, figsize=STYLE["figsize_wide"])
    for digit in range(10):
        U = u_dict[digit]
        for i in range(3):
            img = U[:, i].reshape(28, 28).T
            axes[i, digit].imshow(img, cmap='gray')
            axes[i, digit].axis('off')
            
            # Label digit column on the first row
            if i == 0:
                axes[i, digit].set_title(f"Digit {digit}", fontsize=11, fontweight='bold')
            # Label mode row on the first column
            if digit == 0:
                axes[i, digit].set_ylabel(f"$u_{i+1}$", labelpad=20, rotation=0, va='center', fontsize=12, fontweight='bold')
                axes[i, digit].axis('on')
                axes[i, digit].set_xticks([])
                axes[i, digit].set_yticks([])
                # Remove spine borders but keep ylabel
                for spine in axes[i, digit].spines.values():
                    spine.set_visible(False)
        
    plt.tight_layout()
    plt.savefig("figures/basis_components_grid.png", bbox_inches='tight', dpi=150)
    plt.close()


def plot_reconstruction_comparison(u_dict):
    test_digits = np.load("data/TestDigits.npy")
    test_labels = np.load("data/TestLabels.npy")

    k = 22
    # Pick the first test example for each digit class 0–9
    indices = [np.where(test_labels == d)[0][0] for d in range(10)]

    fig, axes = plt.subplots(2, 10, figsize=STYLE["figsize_wide_short"])

    for col, idx in enumerate(indices):
        digit = test_digits[:, idx]
        label = test_labels[idx]

        # Original
        axes[0, col].imshow(digit.reshape(28, 28).T, cmap="gray")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"{label}", fontsize=11, fontweight='bold')

        # Reconstruction
        Uk = u_dict[label][:, :k]
        recon = Uk @ (Uk.T @ digit)
        axes[1, col].imshow(recon.reshape(28, 28).T, cmap="gray")
        axes[1, col].axis("off")

    # Row labels on the left side
    for row, label_text in enumerate(["Original", f"Rank-{k}"]):
        axes[row, 0].set_ylabel(label_text, labelpad=8, size=11, fontweight='bold')
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/reconstruction_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs("figures", exist_ok=True)
    
    train_digits = np.load("data/TrainDigits.npy")
    train_labels = np.load("data/TrainLabels.npy")
    
    sv_dict = {}
    u_dict = {}
    
    for digit in range(10):
        # Using 400 samples as per original setup
        indices = np.where(train_labels == digit)[0][:400]
        U, S, _ = np.linalg.svd(train_digits[:, indices], full_matrices=False)
        sv_dict[digit] = S
        u_dict[digit] = U
        


    plot_singular_values_comparison(sv_dict)
    plot_all_singular_images(u_dict)
    plot_reconstruction_comparison(u_dict)
    
    # Save singular values for use in classification script
    np.save("data/SingularValues.npy", sv_dict)

if __name__ == "__main__":
    main()
