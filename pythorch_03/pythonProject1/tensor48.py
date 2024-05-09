import torch
from torch import nn

class CustomModel(nn.Module):
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("../models/model.pt", map_location=device)
print(model)