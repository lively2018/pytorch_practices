import torch
import torch.nn as nn
import torch.cuda
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

from resent_501 import ResNet

transforms = Compose(
    [
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    ]
)
training_data = CIFAR10(root='./data', train=True, download=True, transform=transforms)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_classes=10)
model.to(device)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)
num_epochs = 30
for epoch in range(num_epochs):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    iterator.set_description(f"epoch:{epoch+1:4d} loss: {loss.item():.4f}")

torch.save(model.state_dict(), "ResNet.pth")