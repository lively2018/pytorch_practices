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
model = CustomModel().to(device)

model_state_dict = torch.load("../models/model_state_dic.pt", map_location=device)
model.load_state_dict(model_state_dict)

with torch.no_grad():
    model.eval()
    inputs = torch.tensor(
        [
            [1 ** 2, 1],
            [5 ** 2, 5],
            [11 ** 2, 11]
        ], dtype=torch.float, device=device
    ).to(device)
    outputs = model(inputs)
