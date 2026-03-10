import numpy as np
from scipy.linalg import subspace_angles

def extract_gap_vector(mean_residual_map, threshold_percentile=85):
    """
    Extracts a unit vector supported on the high-residual pixels.
    
    Args:
        mean_residual_map: shape (28, 28) or (784,), output from Stage 2.
        threshold_percentile: percentile to threshold the residuals at.
        
    Returns:
        v: (784,) unit vector.
    """
    flat = mean_residual_map.flatten()
    threshold = np.percentile(flat, threshold_percentile)
    mask = (flat > threshold).astype(float)
    v = mask / np.linalg.norm(mask)
    return v

def projection_energy(U_k, v):
    """
    Calculates how much of vector v is captured by the subspace U_k.
    
    Args:
        U_k: (784, k) orthonormal basis.
        v: (784,) unit vector.
        
    Returns:
        rho: scalar in [0, 1].
    """
    proj = U_k @ (U_k.T @ v)
    rho = np.dot(proj, proj)
    return rho

def substitute_basis(U_k, v):
    """
    Replaces the least energetic basis vector in U_k with the component 
    of v orthogonal to the existing subspace.
    
    Args:
        U_k: (784, k) original orthonormal basis.
        v: (784,) gap direction unit vector.
        
    Returns:
        U_k_plus: (784, k) modified subspace.
    """
    proj = U_k @ (U_k.T @ v)
    v_perp = v - proj
    
    if np.linalg.norm(v_perp) < 1e-10:
        raise ValueError("v is already in the subspace — no substitution possible.")
        
    v_perp = v_perp / np.linalg.norm(v_perp)
    
    U_k_plus = U_k.copy()
    U_k_plus[:, -1] = v_perp # Replace the last column (least energetic)
    
    return U_k_plus

def compute_principal_angles(U_1, U_2):
    """
    Computes the principal angles between two subspaces.
    
    Args:
        U_1: (d, k) orthonormal basis for subspace 1.
        U_2: (d, k) orthonormal basis for subspace 2.
        
    Returns:
        angles: Array of principal angles in radians.
    """
    return subspace_angles(U_1, U_2)
