import numpy as np

def load_mnist_data(data_dir="data"):
    """Load and return MNIST training/test sets."""
    train_digits = np.load(f"{data_dir}/TrainDigits.npy")
    train_labels = np.load(f"{data_dir}/TrainLabels.npy")
    test_digits = np.load(f"{data_dir}/TestDigits.npy")
    test_labels = np.load(f"{data_dir}/TestLabels.npy")
    return train_digits, train_labels, test_digits, test_labels

def compute_digit_bases(train_data, labels, num_samples=400, rank_limit=50):
    """Compute SVD bases for each of the 10 digit classes."""
    bases = []
    for d in range(10):
        # Use a subset of training samples for efficiency
        digit_subset = train_data[:, labels == d][:, :num_samples]
        U, _, _ = np.linalg.svd(digit_subset, full_matrices=False)
        bases.append(U[:, :rank_limit])
    return bases

def get_projection_residual(image, basis, k, mean=None):
    """
    Calculate reconstruction residual.
    Supports optional mean subtraction for PCA mode.
    """
    centered = image - mean if mean is not None else image
    Uk = basis[:, :k]
    # Projected vector in the subspace
    projection = Uk @ (Uk.T @ centered)
    return np.linalg.norm(centered - projection)

def get_confusion_rates(true_labels, predicted_labels):
    """Calculate normalized confusion matrix as percentages."""
    conf = np.zeros((10, 10))
    for t, p in zip(true_labels, predicted_labels):
        conf[int(t)][int(p)] += 1
    # Normalize by the number of true instances in each class
    return (conf / conf.sum(axis=1)[:, np.newaxis]) * 100
