from dataset import DatasetArtifact
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader 

data = DatasetArtifact()
X,Y = data.load_dataset()

# label encoding the y features
le = LabelEncoder()
y = le.fit_transform(Y)

# scaling the input features
scaler = StandardScaler()
x = scaler.fit_transform(X)
           # tensor conversion
x = torch.tensor(x, dtype=torch.float32)
y= torch.tensor(y, dtype=torch.long)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

train_dataset =TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(y_train, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

@dataclass
class PreprocessedArtifact:
    train_loader = train_loader
    test_loader = test_loader 

    def load_processed_data(self):
        return train_loader, test_loader