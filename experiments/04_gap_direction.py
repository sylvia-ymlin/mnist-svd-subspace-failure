import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.svd_classifier import SVDRankKClassifier
from src.subspace_analysis import extract_gap_vector, projection_energy

def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading data for digits 3 and 8...")
    train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
    X_train, y_train, _, _ = get_flat_numpy_arrays(train_dataset, test_dataset)
    
    # 1. Compute mean residual maps for digits 3 and 8 at a high enough rank (e.g., k=20)
    print("Computing mean residual maps for k=20 to extract gap vectors...")
    k_res = 20
    model = SVDRankKClassifier(k=k_res)
    model.fit(X_train, y_train)
    
    mean_residuals = {}
    for c in [3, 8]:
        X_c = X_train[y_train == c]
        U_k = model.subspaces_[c]
        P = U_k @ U_k.T
        residuals = X_c - (X_c @ P)
        mean_residuals[c] = np.mean(residuals ** 2, axis=0)
        
    # 2. Extract gap vectors
    v_3 = extract_gap_vector(mean_residuals[3], threshold_percentile=85)
    v_8 = extract_gap_vector(mean_residuals[8], threshold_percentile=85)
    
    # Visualize gap masks
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(v_3.reshape(28, 28), cmap='hot')
    axes[0].set_title('Digit 3 Gap Mask (v_3)')
    axes[0].axis('off')
    axes[1].imshow(v_8.reshape(28, 28), cmap='hot')
    axes[1].set_title('Digit 8 Local Mask (v_8)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('figures/stage4a_gap_masks.png')
    plt.close()
    
    # 3. Compute projection energies across ranks
    k_values = [5, 10, 20, 40]
    results = []
    
    # Pre-generate random vectors for baseline
    np.random.seed(42)
    n_random = 100
    random_vectors = np.random.randn(n_random, 784)
    random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
    
    for k in k_values:
        print(f"Computing energies for k={k}...")
        model_k = SVDRankKClassifier(k=k)
        model_k.fit(X_train, y_train)
        
        U_3 = model_k.subspaces_[3]
        U_8 = model_k.subspaces_[8]
        
        rho_3 = projection_energy(U_3, v_3)
        rho_8 = projection_energy(U_8, v_8)
        
        rho_rand_vals = [projection_energy(U_3, rv) for rv in random_vectors]
        rho_random_mean = np.mean(rho_rand_vals)
        
        results.append({
            'k': k,
            'rho_3': rho_3,
            'rho_8': rho_8,
            'rho_random_mean': rho_random_mean
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/stage4a_rho_values.csv', index=False)
    print(df_results)
    
    # 4. Plot projection energies
    plt.figure(figsize=(6, 4))
    plt.plot(df_results['k'], df_results['rho_3'], marker='o', label='$\\rho_3$ (Digit 3 gap in U_3)')
    plt.plot(df_results['k'], df_results['rho_8'], marker='s', label='$\\rho_8$ (Digit 8 mask in U_8)')
    plt.plot(df_results['k'], df_results['rho_random_mean'], marker='^', linestyle='--', label='$\\rho_{random}$ (Random in U_3)')
    
    plt.xlabel('Rank k')
    plt.ylabel('Projection Energy $\\rho$')
    plt.title('Gap Direction Projection Energy vs Rank')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig('figures/stage4a_projection_energy.png')
    plt.close()
    
    print("Experiment 4a complete. Check figures/stage4a_projection_energy.png for H_A support.")

if __name__ == '__main__':
    main()
