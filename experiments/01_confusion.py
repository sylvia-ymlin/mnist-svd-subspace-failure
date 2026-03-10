import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_filtered_mnist, get_flat_numpy_arrays
from src.svd_classifier import SVDRankKClassifier, SVDFullRankClassifier
from src.baselines import get_logistic_regression
from src.cnn_model import CNNClassifier

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    os.makedirs('figures', exist_ok=True)
    
    print("Loading data for digits 3 and 8...")
    train_dataset, test_dataset = get_filtered_mnist(digits=(3, 8))
    X_train, y_train, X_test, y_test = get_flat_numpy_arrays(train_dataset, test_dataset)
    labels = [3, 8]
    
    # SVD rank-k
    for k in [5, 10, 20, 40]:
        print(f"Training SVD rank-{k}...")
        model = SVDRankKClassifier(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, labels, f'SVD rank-{k} Confusion Matrix', f'figures/confusion_svd_rank_{k}.png')
        
    # SVD full-rank (nearest centroid)
    print("Training SVD full-rank (nearest centroid)...")
    full_svd = SVDFullRankClassifier()
    full_svd.fit(X_train, y_train)
    y_pred_full = full_svd.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_full, labels, 'SVD Full-Rank Confusion Matrix', 'figures/confusion_svd_full.png')
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = get_logistic_regression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_lr, labels, 'Logistic Reg Confusion Matrix', 'figures/confusion_lr.png')
    
    # CNN
    print("Training CNN...")
    cnn = CNNClassifier(epochs=1, batch_size=64)  # 1 epoch is usually enough to shatter simple tasks like 3 vs 8
    cnn.fit(train_dataset)
    y_pred_cnn = cnn.predict(test_dataset)
    plot_confusion_matrix(y_test, y_pred_cnn, labels, 'CNN Confusion Matrix', 'figures/confusion_cnn.png')
    print("Experiment 1 complete.")

if __name__ == '__main__':
    main()
