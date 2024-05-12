import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

import tqdm
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torch.utils.data import DataLoader

from torch.optim.adam import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vgg16(pretrained=True)
fc = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(4096, 10),
)
model.classifier = fc
model.to(device)

transforms = Compose(
    [
        Resize(224),
        RandomCrop((224, 224), padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]
)

training_data = CIFAR10(root="./data", train=True, transform=transforms, download=True)
test_data = CIFAR10(root="./data", train=False, transform=transforms, download=True)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss.item()}")

torch.save(model.state_dict(), "CIFAR_pretrained.pth")
