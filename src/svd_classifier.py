import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SVDRankKClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=10):
        self.k = k
        self.subspaces_ = {} # class_label -> U_k (d x k)
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            X_c = X[y == c]
            # SVD on transpose: X_c is (N, d), so X_c.T is (d, N).
            # Columns of U are left singular vectors (basis for feature space).
            U, s, Vt = np.linalg.svd(X_c.T, full_matrices=False)
            self.subspaces_[c] = U[:, :self.k]
        return self
        
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i]
            best_class = None
            max_energy = -np.inf
            
            for c in self.classes_:
                U_k = self.subspaces_[c]
                # Projection energy: || U_k^T x ||^2
                proj_energy = np.sum((U_k.T @ x) ** 2)
                if proj_energy > max_energy:
                    max_energy = proj_energy
                    best_class = c
            predictions.append(best_class)
        return np.array(predictions)
        
class SVDFullRankClassifier(BaseEstimator, ClassifierMixin):
    """
    SVD full-rank classifier (nearest centroid) to isolate the effect of rank truncation.
    """
    def __init__(self):
        self.centroids_ = {}
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.centroids_[c] = np.mean(X[y == c], axis=0)
        return self
        
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i]
            best_class = None
            min_dist = np.inf
            for c in self.classes_:
                dist = np.linalg.norm(x - self.centroids_[c])
                if dist < min_dist:
                    min_dist = dist
                    best_class = c
            predictions.append(best_class)
        return np.array(predictions)
