import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.svd_classifier import SVDRankKClassifier
from src.subspace_analysis import extract_gap_vector, substitute_basis, compute_principal_angles

def plot_confusion_comparison(cm_orig, cm_mod, labels, title_orig, title_mod, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(title_orig)
    
    sns.heatmap(cm_mod, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title(title_mod)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading data for digits 3 and 8...")
    train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
    X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_dataset, test_dataset)
    labels = [3, 8]
    
    # 1. Get gap vector from k=20 residuals (consistent with Stage 4a)
    model_res = SVDRankKClassifier(k=20)
    model_res.fit(X_train, y_train)
    X_3 = X_train[y_train == 3]
    U_20 = model_res.subspaces_[3]
    P = U_20 @ U_20.T
    residuals_3 = X_3 - (X_3 @ P)
    mean_res_3 = np.mean(residuals_3 ** 2, axis=0)
    v_3 = extract_gap_vector(mean_res_3, threshold_percentile=85)
    
    # 2. Re-run and substitute for k={5, 10, 20}
    k_values = [5, 10, 20]
    results = []
    
    for k in k_values:
        print(f"Running substitution for k={k}...")
        # Original model
        model = SVDRankKClassifier(k=k)
        model.fit(X_train, y_train)
        y_pred_orig = model.predict(X_test)
        cm_orig = confusion_matrix(y_test, y_pred_orig, labels=labels)
        
        # Determine 3->8 errors (row 0, col 1 since labels=[3, 8])
        err_3_8_orig = cm_orig[0, 1]
        
        # Modified model (intervene on subspace 3)
        model_mod = SVDRankKClassifier(k=k)
        model_mod.fit(X_train, y_train) # Get base subspaces
        
        # Save original subspace for comparison
        U_orig_3 = model_mod.subspaces_[3].copy()
        
        # Substitute basis for digit 3
        try:
            model_mod.subspaces_[3] = substitute_basis(model_mod.subspaces_[3], v_3)
            
            # Compute principal angles to measure manifold rotation
            angles = compute_principal_angles(U_orig_3, model_mod.subspaces_[3])
            mean_angle_deg = np.degrees(np.mean(angles))
            print(f"Mean principal angle (rotation) for k={k}: {mean_angle_deg:.2f} degrees")
            
            y_pred_mod = model_mod.predict(X_test)
            cm_mod = confusion_matrix(y_test, y_pred_mod, labels=labels)
            
            err_3_8_mod = cm_mod[0, 1]
            delta = err_3_8_orig - err_3_8_mod
            
            # Plot comparison
            plot_confusion_comparison(
                cm_orig, cm_mod, labels, 
                f'Original Rank-{k}', f'Modified Rank-{k}', 
                f'figures/stage5_confusion_comparison_k{k}.png'
            )
            
        except ValueError as e:
            print(f"Skipping substitution for k={k}: {e}")
            err_3_8_mod = np.nan
            delta = np.nan
            
        results.append({
            'k': k,
            'err_3_8_orig': err_3_8_orig,
            'err_3_8_mod': err_3_8_mod,
            'delta_reduction': delta,
            'mean_angle_deg': mean_angle_deg
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/stage5_confusion_delta.csv', index=False)
    print(df_results)
    
    # 3. Plot delta vs rank
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in df_results['k']], df_results['delta_reduction'], color='skyblue')
    plt.xlabel('Rank k')
    plt.ylabel('Reduction in 3->8 Errors (Delta)')
    plt.title('Effect of Basis Substitution on Confusion')
    plt.tight_layout()
    plt.savefig('figures/stage5_delta_vs_rank.png')
    plt.close()
    
    print("Experiment 5 complete. Check figures/ and results/ for outcome.")

if __name__ == '__main__':
    main()
