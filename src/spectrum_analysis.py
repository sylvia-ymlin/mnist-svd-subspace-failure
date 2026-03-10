import numpy as np

def compute_centered_spectrum(X_c):
    """
    Computes the normalized cumulative energy curve for a single class.
    
    Args:
        X_c: array of shape (N, d) containing data for class c.
        
    Returns:
        f_k: array of normalized cumulative energies for k=1..d
    """
    # Center the data
    mean_X = np.mean(X_c, axis=0)
    X_centered = X_c - mean_X
    
    # Compute SVD
    # X_centered.T is (d, N), SVD gives U (d, d), s (min(N, d))
    U, s, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
    
    # Energy is proportional to square of singular values
    energy = s ** 2
    total_energy = np.sum(energy)
    
    # Cumulative energy
    cumulative_energy = np.cumsum(energy)
    
    # Normalize
    f_k = cumulative_energy / total_energy
    return f_k
