import torch
from cnn_406 import CNN
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import RandomCrop, Compose, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.datasets.cifar import CIFAR10

model = CNN(num_classes=10)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load("CIFAR.pth", map_location=device))

num_corr = 0

transforms = Compose(
    [
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))


    ]
)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy: {num_corr / len(test_data)}")
