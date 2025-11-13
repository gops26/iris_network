import torch.nn as nn

class IrisNetworkModel(nn.Module):
    def __init__(self):
        super(IrisNetworkModel, self).__init__()
        self.fc1 = nn.Linear(4,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x