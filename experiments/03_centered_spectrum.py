import os
import sys
import matplotlib.pyplot as plt

# ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.spectrum_analysis import compute_centered_spectrum

def main():
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data for digits 3 and 8...")
    train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
    X_train, y_train, _, _ = get_flat_numpy_arrays(train_dataset, test_dataset)
    
    f_k_dict = {}
    
    print("Computing centered spectra...")
    for c in [3, 8]:
        X_c = X_train[y_train == c]
        f_k = compute_centered_spectrum(X_c)
        f_k_dict[c] = f_k
        
    plt.figure(figsize=(8, 6))
    for c in [3, 8]:
        # Plot up to first 200 components
        max_k = min(200, len(f_k_dict[c]))
        plt.plot(range(1, max_k + 1), f_k_dict[c][:max_k], label=f'Digit {c}')
        
    plt.xlabel('Rank k')
    plt.ylabel('Normalized Cumulative Energy')
    plt.title('Centered Spectrum Comparison (Intrinsic Dimensionality)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/centered_spectrum.png')
    plt.close()
    
    print("Difference at k=20:")
    print(f"Digit 3: {f_k_dict[3][19]:.4f}")
    print(f"Digit 8: {f_k_dict[8][19]:.4f}")
    
    print("Experiment 3 complete.")

if __name__ == '__main__':
    main()
