import torch
from torch import nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("../models/model.pt", map_location=device)
print(model)

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