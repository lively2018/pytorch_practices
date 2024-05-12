import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor



training_data = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())

imgs = [item[0] for item in training_data]

imgs = torch.stack(imgs, dim=0).numpy()

mean_r = imgs[:, 0, :, :].mean()
mean_g = imgs[:, 1, :, :].mean()
mean_b = imgs[:, 2, :, :].mean()

print(mean_r, mean_g, mean_b)

std_r = imgs[:, 0, :, :].std()
std_g = imgs[:, 1, :, :].std()
std_b = imgs[:, 2, :, :].std()
print(std_r, std_g, std_b)