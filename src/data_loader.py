import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

def get_filtered_mnist(digits=None, data_dir="./data", download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform
    )

    if digits is not None:
        train_indices = [i for i, targ in enumerate(train_dataset.targets) if int(targ) in digits]
        test_indices = [i for i, targ in enumerate(test_dataset.targets) if int(targ) in digits]

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, test_dataset

def get_flat_numpy_arrays(train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    def to_numpy(loader):
        for x, y in loader:
            return x.view(x.size(0), -1).numpy(), y.numpy()

    X_train, y_train = to_numpy(train_loader)
    X_test, y_test = to_numpy(test_loader)

    return X_train, y_train, X_test, y_test
