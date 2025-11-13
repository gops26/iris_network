from model import IrisNetworkModel
import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import PreprocessedArtifact

dataset = PreprocessedArtifact()
train_loader, test_loader = dataset.load_processed_data()

m1 = IrisNetworkModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m1.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    m1.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = m1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} / {num_epochs} epoch loss = {epoch_loss}")

torch.save(m1.state_dict(), "iris_model.pth")