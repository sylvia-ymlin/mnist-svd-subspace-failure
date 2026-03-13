class CNNClassifier:
    """A simple CNN baseline for MNIST."""
    def __init__(self, epochs=5, lr=0.001, batch_size=64):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None

    def fit(self, train_dataset):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model(data), target)
                loss.backward()
                optimizer.step()
        return self
