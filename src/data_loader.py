import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np

def get_filtered_mnist(digits=(3, 8), data_dir='./data', download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
    
    if digits is not None:
        train_indices = [i for i, targ in enumerate(train_dataset.targets) if targ in digits]
        test_indices = [i for i, targ in enumerate(test_dataset.targets) if targ in digits]
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
    return train_dataset, test_dataset

def get_flat_numpy_arrays(train_dataset, test_dataset, batch_size=None):
    if batch_size is None:
        batch_size = max(len(train_dataset), len(test_dataset))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    X_train_tensor, y_train_tensor = next(iter(train_loader))
    X_test_tensor, y_test_tensor = next(iter(test_loader))
    
    X_train = X_train_tensor.view(X_train_tensor.shape[0], -1).numpy()
    y_train = y_train_tensor.numpy()
    
    X_test = X_test_tensor.view(X_test_tensor.shape[0], -1).numpy()
    y_test = y_test_tensor.numpy()
    
    return X_train, y_train, X_test, y_test
