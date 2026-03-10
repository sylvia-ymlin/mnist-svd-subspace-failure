import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.svd_classifier import SVDRankKClassifier

def plot_heatmap(heatmap, title, filename):
    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap.reshape(28, 28), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data for digits 3 and 8...")
    train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
    X_train, y_train, _, _ = get_flat_numpy_arrays(train_dataset, test_dataset)
    
    k_values = [10, 20]
    
    for k in k_values:
        print(f"Computing residuals for SVD rank-{k}...")
        model = SVDRankKClassifier(k=k)
        model.fit(X_train, y_train)
        
        for c in [3, 8]:
            X_c = X_train[y_train == c]
            U_k = model.subspaces_[c]
            
            # Projection P = U_k U_k^T
            # Residual R = x - P x
            P = U_k @ U_k.T
            X_c_proj = X_c @ P
            residuals = X_c - X_c_proj
            
            # Mean residual map (mean of squared residuals to act as energy)
            mean_residual_map = np.mean(residuals ** 2, axis=0)
            
            plot_heatmap(mean_residual_map, f'Mean squared residual for digit {c} (rank-{k})', f'figures/residual_heatmap_c{c}_k{k}.png')
    
    print("Experiment 2 complete.")

if __name__ == '__main__':
    main()
