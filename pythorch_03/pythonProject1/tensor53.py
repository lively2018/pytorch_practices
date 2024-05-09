import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.tensor([self.x1[index], self.x2[index], self.x3[index]], dtype=torch.float)
        y = torch.tensor([self.y[index]], dtype=torch.float)
        return x, y

    def __len__(self):
        return self.length

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

dataset = CustomDataset("../../../../dataset/pytorch_practice/datasets/binary.csv")
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
validation_size = int(0.1 * dataset_size)
test_size = int(0.1 * dataset_size)

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], torch.manual_seed(4))
train_dataload = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
validation_dataload = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataload = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel()
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataload:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    cost = cost / len(train_dataload)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}")

with torch.no_grad():
    model.eval()
    for x, y in validation_dataload:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        print(outputs)
        print(outputs >= torch.tensor([0.5], dtype=torch.float, device=device))
        print("----------------------------")