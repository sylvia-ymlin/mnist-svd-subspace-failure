import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from style_utils import STYLE, apply_global_style

apply_global_style()

def classify(bases, means, test_digits, k):
    """
    Classify test_digits using projection residuals.
    If means is not None, subtract class mean before projecting (PCA mode).
    """
    dists = []
    for d in range(10):
        Uk = bases[d][:, :k]
        if means is not None:
            centered = test_digits - means[d][:, np.newaxis]
        else:
            centered = test_digits
        proj = Uk @ (Uk.T @ centered)
        dists.append(np.linalg.norm(centered - proj, axis=0))
    return np.argmin(np.array(dists), axis=0)

def confusion_matrix_normalized(preds, labels):
    conf = np.zeros((10, 10))
    for t, p in zip(labels, preds):
        conf[int(t)][int(p)] += 1
    return conf / conf.sum(axis=1, keepdims=True) * 100

def plot_accuracy_vs_rank(ks, svd_accs, pca_accs, best_k_svd, best_k_pca):
    fig, ax = plt.subplots(figsize=STYLE["figsize_single"])
    ax.plot(ks, svd_accs, color=STYLE["colors"]["primary"], linewidth=2,
            marker="o", markersize=3, markerfacecolor="white",
            markeredgewidth=1.2, markeredgecolor=STYLE["colors"]["primary"],
            label="SVD (uncentered)", zorder=2)
    ax.plot(ks, pca_accs, color=STYLE["colors"]["secondary"], linewidth=2,
            marker="s", markersize=3, markerfacecolor="white",
            markeredgewidth=1.2, markeredgecolor=STYLE["colors"]["secondary"],
            label="PCA (centered)", zorder=2)

    # Mark peaks
    ax.plot(best_k_svd, max(svd_accs), "o", markersize=9,
            color=STYLE["colors"]["primary"], zorder=3)
    ax.plot(best_k_pca, max(pca_accs), "s", markersize=9,
            color=STYLE["colors"]["secondary"], zorder=3)

    svd_offset_y = -0.8
    pca_offset_y = +0.5
    if abs(best_k_svd - best_k_pca) < 5:
        svd_offset_y = -1.2
        pca_offset_y = +0.8

    ax.annotate(f"{max(svd_accs):.2f}% (k={best_k_svd})",
                xy=(best_k_svd, max(svd_accs)),
                xytext=(best_k_svd + 4, max(svd_accs) + svd_offset_y),
                fontsize=9, color=STYLE["colors"]["primary"],
                arrowprops=dict(arrowstyle="->", color=STYLE["colors"]["primary"], lw=1))
    ax.annotate(f"{max(pca_accs):.2f}% (k={best_k_pca})",
                xy=(best_k_pca, max(pca_accs)),
                xytext=(best_k_pca + 4, max(pca_accs) + pca_offset_y),
                fontsize=9, color=STYLE["colors"]["secondary"],
                arrowprops=dict(arrowstyle="->", color=STYLE["colors"]["secondary"], lw=1))

    ax.set_xlabel("Subspace rank $k$")
    ax.set_ylabel("Classification accuracy (%)")
    ax.legend()
    plt.savefig("figures/pca_accuracy_vs_rank.png")
    plt.close()

def plot_confusion_comparison(conf_svd, conf_pca):
    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_pair"])
    ims = []
    for ax, conf in zip(axes, [conf_svd, conf_pca]):
        im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=100)
        ims.append(im)
        for i in range(10):
            for j in range(10):
                val = conf[i, j]
                color = "white" if val > 50 else STYLE["colors"]["text"]
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")

    fig.colorbar(ims[-1], ax=axes, shrink=0.82, label="Rate (%)")
    plt.savefig("figures/pca_confusion_comparison.png")
    plt.close()

def main():
    train_digits = np.load("data/TrainDigits.npy")
    train_labels = np.load("data/TrainLabels.npy")
    test_digits  = np.load("data/TestDigits.npy")
    test_labels  = np.load("data/TestLabels.npy")

    # --- Build SVD and PCA bases ---
    bases_svd = []
    bases_pca = []
    means_pca = []
    for d in range(10):
        X = train_digits[:, train_labels == d][:, :400]
        # Uncentered SVD (V0)
        U_svd, _, _ = np.linalg.svd(X, full_matrices=False)
        bases_svd.append(U_svd)
        # Centered SVD = PCA (V4)
        mu = X.mean(axis=1)
        U_pca, _, _ = np.linalg.svd(X - mu[:, np.newaxis], full_matrices=False)
        bases_pca.append(U_pca)
        means_pca.append(mu)

    # --- k sweep ---
    ks = range(1, 51)
    svd_accs = []
    pca_accs = []
    for k in ks:
        preds_svd = classify(bases_svd, None,      test_digits, k)
        preds_pca = classify(bases_pca, means_pca, test_digits, k)
        svd_accs.append(np.mean(preds_svd == test_labels) * 100)
        pca_accs.append(np.mean(preds_pca == test_labels) * 100)

    best_k_svd = list(ks)[np.argmax(svd_accs)]
    best_k_pca = list(ks)[np.argmax(pca_accs)]
    
    # --- Confusion matrices at best k ---
    preds_svd_best = classify(bases_svd, None,      test_digits, best_k_svd)
    preds_pca_best = classify(bases_pca, means_pca, test_digits, best_k_pca)
    conf_svd = confusion_matrix_normalized(preds_svd_best, test_labels)
    conf_pca = confusion_matrix_normalized(preds_pca_best, test_labels)

    # --- Plots ---
    plot_accuracy_vs_rank(list(ks), svd_accs, pca_accs, best_k_svd, best_k_pca)
    plot_confusion_comparison(conf_svd, conf_pca)

if __name__ == "__main__":
    main()
