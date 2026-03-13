"""
Experiment 04: Robustness benchmark.
Gaussian and Salt-and-Pepper noise curves for SVD (k=15) vs CNN.

Run from repo root:
    python experiments/04_robustness_benchmark.py

Output: figures/robustness_combined.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.plotting import set_plot_theme, save_figure, NBody_Palette
from src.cnn_model import CNNClassifier

# Evaluation grids
GAUSSIAN_NOISE = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
SP_NOISE = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
K_SVD = 15

DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

set_plot_theme()

def main():
    train_ds, test_ds = get_filtered_mnist(digits=None, data_dir=str(DATA_DIR))
    X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_ds, test_ds)
    
    classes = np.arange(10)
    
    # ── Fit SVD subspaces ──────────────────────────────────────────────────────────
    print("Computing SVD subspaces...")
    U_full = {}
    for c in classes:
        X_c = X_train[y_train == c]
        U, _, _ = np.linalg.svd(X_c.T, full_matrices=False)
        U_full[c] = U

    def svd_predict(X_noisy, k):
        scores = np.zeros((len(X_noisy), 10))
        for c in classes:
            U_k_c = U_full[c][:, :k]
            scores[:, c] = np.sum((X_noisy @ U_k_c) ** 2, axis=1)
        return scores.argmax(axis=1)

    # ── Fit PCA+LR baseline (same k=15 components, global subspace) ──────────────
    print("Fitting PCA+LR baseline (k=15 components)...")
    pca = PCA(n_components=K_SVD)
    pca.fit(X_train)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(pca.transform(X_train), y_train)
    pca_clean_acc = lr.score(pca.transform(X_test), y_test)
    print(f"  PCA+LR clean accuracy: {pca_clean_acc:.4f}")

    def pca_lr_predict(X_noisy):
        return lr.predict(pca.transform(X_noisy))

    # ── Train CNN on clean data ───────────────────────────────────────────────────
    print("Training CNN on clean data (5 epochs)...")
    cnn = CNNClassifier(epochs=5)
    cnn.fit(train_ds)

    def cnn_predict(X_noisy):
        import torch
        device = next(cnn.model.parameters()).device
        cnn.model.eval()
        
        preds = []
        batch_size = 500
        for i in range(0, len(X_noisy), batch_size):
            batch = X_noisy[i:i+batch_size]
            X_tensor = torch.tensor(batch, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
            with torch.no_grad():
                batch_preds = cnn.model(X_tensor).argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
        return np.array(preds)

    # ── Evaluate Gaussian ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    results_gauss = {"SVD k=15": [], "CNN": [], "PCA+LR k=15": []}
    
    print("\nEvaluating Gaussian robustness...")
    for sigma in GAUSSIAN_NOISE:
        noise = rng.normal(0, sigma, X_test.shape)
        X_noisy = X_test + noise  
        
        preds_svd = svd_predict(X_noisy, k=K_SVD)
        acc_svd = np.mean(preds_svd == y_test)
        results_gauss["SVD k=15"].append(acc_svd)

        preds_cnn = cnn_predict(X_noisy)
        acc_cnn = np.mean(preds_cnn == y_test)
        results_gauss["CNN"].append(acc_cnn)

        preds_pca = pca_lr_predict(X_noisy)
        results_gauss["PCA+LR k=15"].append(np.mean(preds_pca == y_test))

    # ── Evaluate Salt and Pepper ──────────────────────────────────────────────────
    results_sp = {"SVD k=15": [], "CNN": [], "PCA+LR k=15": []}

    print("\nEvaluating Salt-and-Pepper robustness...")
    for p in SP_NOISE:
        noise_mask = rng.random(X_test.shape)
        X_noisy = X_test.copy()
        salt = noise_mask < (p / 2.0)
        pepper = (noise_mask >= (p / 2.0)) & (noise_mask < p)

        # Black (0) and white (1) in raw [0,1] pixel space, converted to the
        # same normalized space as X_test via torchvision Normalize((0.1307,),(0.3081,)).
        val_black = (0.0 - 0.1307) / 0.3081   # ≈ -0.424
        val_white = (1.0 - 0.1307) / 0.3081   # ≈  2.821

        X_noisy[pepper] = val_black
        X_noisy[salt]   = val_white

        preds_svd = svd_predict(X_noisy, k=K_SVD)
        acc_svd = np.mean(preds_svd == y_test)
        results_sp["SVD k=15"].append(acc_svd)
        
        preds_cnn = cnn_predict(X_noisy)
        acc_cnn = np.mean(preds_cnn == y_test)
        results_sp["CNN"].append(acc_cnn)

        preds_pca = pca_lr_predict(X_noisy)
        results_sp["PCA+LR k=15"].append(np.mean(preds_pca == y_test))

    # ── Plotting ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Gaussian
    axes[0].plot(GAUSSIAN_NOISE, results_gauss["SVD k=15"], 'o-', color=NBody_Palette["blue_deep"], label='SVD Subspace (k=15)', linewidth=2)
    axes[0].plot(GAUSSIAN_NOISE, results_gauss["CNN"], 's--', color=NBody_Palette["red"], label='CNN (Clean Train)', linewidth=2)
    axes[0].plot(GAUSSIAN_NOISE, results_gauss["PCA+LR k=15"], '^:', color=NBody_Palette["orange"], label='PCA+LR (k=15)', linewidth=2)
    axes[0].set_title("Robustness: Gaussian Noise")
    axes[0].set_xlabel("Noise Std Dev (\u03c3)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # Right subplot: Salt & Pepper
    axes[1].plot(SP_NOISE, results_sp["SVD k=15"], 'o-', color=NBody_Palette["blue_deep"], label='SVD Subspace (k=15)', linewidth=2)
    axes[1].plot(SP_NOISE, results_sp["CNN"], 's--', color=NBody_Palette["red"], label='CNN (Clean Train)', linewidth=2)
    axes[1].plot(SP_NOISE, results_sp["PCA+LR k=15"], '^:', color=NBody_Palette["orange"], label='PCA+LR (k=15)', linewidth=2)
    axes[1].set_title("Robustness: Salt-and-Pepper Noise")
    axes[1].set_xlabel("Noise Probability ($p$)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    save_figure(fig, "robustness_combined.png", [FIGURES_DIR])
    
if __name__ == "__main__":
    main()
