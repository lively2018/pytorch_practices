import torch
from torch import nn
from torch import optim


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        self.layer1[0].weight.data = torch.nn.Parameter(
            torch.tensor([[0.4352, 0.3545],
                         [0.1951, 0.4835]], dtype=torch.float)
        )
        self.layer1[0].bias.data = torch.nn.Parameter(
            torch.tensor([[-0.1415, 0.0439]], dtype=torch.float)
        )
        self.layer2[0].weight.data = torch.nn.Parameter(
            torch.tensor([[-0.1725, 0.1129]], dtype=torch.float)
        )
        self.layer2[0].bias.data = torch.nn.Parameter(
            torch.tensor([[-0.3043]], dtype=torch.float)
        )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)
