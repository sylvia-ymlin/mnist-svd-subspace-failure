import numpy as np
import os
from sklearn.datasets import fetch_openml

def normalize_data(x):
    """L2-normalize image vectors for consistent SVD computation."""
    norms = np.linalg.norm(x, axis=0)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return x / norms

def download_mnist(data_dir):
    """Download MNIST and split into Train/Test sets as NumPy arrays."""
    # Use a local directory for scikit-learn cache to avoid permission errors
    cache_dir = os.path.join(data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', data_home=cache_dir)
    
    # Split into 60,000 training and 10,000 test samples
    X_train, X_test = mnist.data[:60000], mnist.data[60000:]
    y_train, y_test = mnist.target[:60000], mnist.target[60000:]
    
    # Transpose to (784, N) as required by project logic
    X_train_t = X_train.T.astype(np.float32)
    X_test_t  = X_test.T.astype(np.float32)
    
    # Convert labels to integers
    y_train = y_train.astype(np.int32)
    y_test  = y_test.astype(np.int32)
    
    # Normalize images
    print("  Normalizing data...")
    X_train_norm = normalize_data(X_train_t)
    X_test_norm  = normalize_data(X_test_t)
    
    # Save files
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "TrainDigits.npy"), X_train_norm)
    np.save(os.path.join(data_dir, "TrainLabels.npy"), y_train)
    np.save(os.path.join(data_dir, "TestDigits.npy"), X_test_norm)
    np.save(os.path.join(data_dir, "TestLabels.npy"), y_test)
    
    print(f"  Saved to {data_dir}/")
    print(f"  Train: Digits {X_train_norm.shape}, Labels {y_train.shape}")
    print(f"  Test:  Digits {X_test_norm.shape}, Labels {y_test.shape}")

def main():
    """Main data preparation script: downloads and formats MNIST datasets."""
    data_dir = "./data"
    download_mnist(data_dir)

if __name__ == "__main__":
    main()
