import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.transforms import RandomCrop, Compose, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.datasets.cifar import CIFAR10
from cnn_406 import CNN

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

model = CNN(num_classes=10)

model.to(device)

lr = 1e-3

optim = Adam(model.parameters(), lr=lr)

for epoch in range(100):
    for data, label in train_loader:
        optim.zero_grad()

        preds = model(data.to(device))

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    if epoch == 0 or epoch % 10 == 9:
        print(f"Epoch {epoch+1} , loss {loss.item()}")

torch.save(model.state_dict(), 'CIFAR.pth')
