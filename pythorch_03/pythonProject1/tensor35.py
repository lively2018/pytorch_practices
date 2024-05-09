import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.tensor([self.x[index] ** 2, self.x[index]], dtype=torch.float)
        y = torch.tensor([self.y[index]], dtype=torch.float)
        return x, y

    def __len__(self):
        return self.length

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

train_dataset = CustomDataset("../../../../dataset/pytorch_practice/datasets/non_linear.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(1000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    cost = cost / len(train_dataloader)
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch+1:4d}, Model : {list(model.parameters())}, Cost: {cost:.3f}")

with torch.no_grad():
    model.eval()

    inputs = torch.tensor(
        [
            [1 ** 2, 1],
            [5 ** 2, 5],
            [11 ** 2, 11]
        ]
    , dtype=torch.float).to(device)

    outputs = model(inputs)
    print(outputs)
torch.save(model,
           "../models/model.pt")
torch.save(model.state_dict(),
           "../models/model_state_dic.pt")