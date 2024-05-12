import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torch.utils.data import DataLoader

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

model.load_state_dict(torch.load("CIFAR_pretrained.pth", map_location=device))

num_corr = 0
test_data = CIFAR10(root="./data", train=False, transform=transforms, download=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr
    print(f"Accuracy: {num_corr/len(test_data)}")
