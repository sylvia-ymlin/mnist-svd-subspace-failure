import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from style_utils import STYLE, apply_global_style

apply_global_style()

def plot_overlap_heatmap(overlap_matrix):
    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    masked = overlap_matrix.copy()
    np.fill_diagonal(masked, np.nan)
    
    im = ax.imshow(masked, cmap="YlGnBu", vmin=np.nanmin(masked), vmax=np.nanmax(masked))
    for i in range(10):
        for j in range(10):
            if i == j: 
                ax.text(j, i, "—", ha="center", va="center", color="#999")
            else:
                val = overlap_matrix[i, j]
                color = "white" if val > np.nanmean(masked) else STYLE["colors"]["text"]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                        fontsize=8.5, color=color)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Digit class $j$")
    ax.set_ylabel("Digit class $i$")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Frobenius Overlap $\|U_i^T U_j\|_F^2$")
    plt.savefig("figures/subspace_overlap_heatmap.png")
    plt.close()

def plot_stability(variations):
    """
    variations: list of 10 lists, each containing (min, mean, max) tuples
                from bootstrap resamples for one digit class.
    Three subplots show min / mean / max principal angle distributions,
    revealing whether stability holds only for the best-aligned direction
    or across the full subspace.
    """
    mins  = [[v[0] for v in d] for d in variations]
    means = [[v[1] for v in d] for d in variations]
    maxs  = [[v[2] for v in d] for d in variations]

    labels = [str(i) for i in range(10)]
    titles = ["Min principal angle (best-aligned direction)",
              "Mean principal angle (average across subspace)",
              "Max principal angle (worst-aligned direction)"]
    colors = [STYLE["colors"]["primary"], STYLE["colors"]["accent"], STYLE["colors"]["secondary"]]

    fig, axes = plt.subplots(3, 1, figsize=STYLE["figsize_portrait"], sharex=True)
    for ax, data, title, color in zip(axes, [mins, means, maxs], titles, colors):
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(title)

    axes[-1].set_xlabel("Digit class")
    fig.suptitle("Bootstrap Subspace Stability (300/400 samples, 10 resamples)")
    plt.tight_layout()
    plt.savefig("figures/stability_analysis.png")
    plt.close()


def main():
    # Load data
    train_digits = np.load("data/TrainDigits.npy")
    train_labels = np.load("data/TrainLabels.npy")
    test_digits = np.load("data/TestDigits.npy")
    test_labels = np.load("data/TestLabels.npy")
    
    # Use global optimal k=22 for geometric comparisons
    k_ref = 22
    bases = []
    means = []
    for d in range(10):
        digit_data = train_digits[:, train_labels == d][:, :400]
        means.append(np.mean(digit_data, axis=1))
        U, _, _ = np.linalg.svd(digit_data, full_matrices=False)
        bases.append(U[:, :k_ref])

    # 1. Pairwise Frobenius Overlap
    overlap_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i == j: continue
            # Sum of squared cosines of principal angles
            S = np.linalg.svd(bases[i].T @ bases[j], compute_uv=False)
            overlap_matrix[i, j] = np.sum(S**2)

    # 2. Bootstrap Stability Test
    stability_variations = []
    for d in range(10):
        digit_full = train_digits[:, train_labels == d][:, :400]
        U_orig = bases[d]
        digit_variations = []
        for _ in range(10):
            idx = np.random.choice(400, 300, replace=False)
            U_boot, _, _ = np.linalg.svd(digit_full[:, idx], full_matrices=False)
            U_boot_k = U_boot[:, :k_ref]
            # Full principal angle sequence between original and bootstrap subspace
            S = np.linalg.svd(U_orig.T @ U_boot_k, compute_uv=False)
            angles = np.degrees(np.arccos(np.clip(S, 0, 1)))
            digit_variations.append((np.min(angles), np.mean(angles), np.max(angles)))
        stability_variations.append(digit_variations)

    # 3. Multiple Regression Analysis
    # Get confusion from classification script results
    best_preds = np.load("data/BestPredictions.npy")
    conf = np.zeros((10, 10))
    for t, p in zip(test_labels, best_preds):
        conf[t][p] += 1
    
    X_reg = []
    y_reg = []
    for i in range(10):
        for j in range(10):
            if i == j: continue
            # Predictors: Overlap, Mean Distance
            overlap = overlap_matrix[i, j]
            mean_dist = np.linalg.norm(means[i] - means[j])
            
            # Target: Confusion rate i -> j (%)
            confusion_rate = conf[i, j] / np.sum(test_labels == i) * 100
            
            X_reg.append([overlap, mean_dist])
            y_reg.append(confusion_rate)
            
    X_reg = np.array(X_reg)
    y_reg = np.array(y_reg)
    
    # Fit regression with sklearn
    # Standardize features before comparing coefficients — overlap and mean_dist
    # are on different scales, so raw coefficients cannot be directly compared.
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    scaler = StandardScaler()
    X_reg_scaled = scaler.fit_transform(X_reg)

    model = LinearRegression()
    model.fit(X_reg_scaled, y_reg)

    r2_insample = model.score(X_reg_scaled, y_reg)
    coeffs = model.coef_   # comparable: both in units of std-dev of predictor

    # LOO R²: aggregate all held-out predictions first, then compute R² once.
    # (sklearn r2_score is undefined for a single test point, so per-fold averaging gives nan.)
    loo = LeaveOneOut()
    y_loo = np.zeros_like(y_reg)
    for train_idx, test_idx in loo.split(X_reg_scaled):
        m = LinearRegression().fit(X_reg_scaled[train_idx], y_reg[train_idx])
        y_loo[test_idx] = m.predict(X_reg_scaled[test_idx])
    ss_res = np.sum((y_reg - y_loo) ** 2)
    ss_tot = np.sum((y_reg - y_reg.mean()) ** 2)
    r2_loo = 1 - ss_res / ss_tot

    # Independent Spearman correlations as a sanity check
    rho_overlap,   p_overlap   = spearmanr(X_reg[:, 0], y_reg)
    rho_mean_dist, p_mean_dist = spearmanr(X_reg[:, 1], y_reg)

    print("\n--- Multiple Regression Results (Confusion ~ Overlap + MeanDist) ---")
    print(f"Features standardized before regression (coefficients are comparable).")
    print(f"R² in-sample:        {r2_insample:.3f}")
    print(f"R² LOO (honest):     {r2_loo:.3f}  (gap = {r2_insample - r2_loo:.3f})")
    print(f"Standardized coefficients: Overlap={coeffs[0]:.3f}, MeanDist={coeffs[1]:.3f}")
    print(f"\nIndependent Spearman correlations:")
    print(f"  Overlap   vs confusion: ρ={rho_overlap:.3f}  (p={p_overlap:.4f})")
    print(f"  Mean dist vs confusion: ρ={rho_mean_dist:.3f}  (p={p_mean_dist:.4f})")


    
    # 4. Visualizations
    plot_overlap_heatmap(overlap_matrix)
    plot_stability(stability_variations)

if __name__ == "__main__":
    main()
