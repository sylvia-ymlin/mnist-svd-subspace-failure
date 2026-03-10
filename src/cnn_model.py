import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNClassifier:
    """Wrapper to act like an sklearn estimator for the experiments."""
    def __init__(self, epochs=5, lr=0.001, batch_size=64, device=None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu' if not torch.backends.mps.is_available() else 'mps')
        self.model = None
        self.classes_ = None
        
    def fit(self, train_dataset):
        self.model = SimpleCNN(num_classes=10).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return self
        
    def predict(self, test_dataset):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy().flatten())
        return np.array(predictions)
