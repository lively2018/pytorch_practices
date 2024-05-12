import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Normalize

transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247,0.243,0.261)),
        T.ToPILImage()
    ]
)
train_data = CIFAR10(root='./data', train=True, transform=transforms, download=True)
test_data = CIFAR10(root='./data', train=False, transform=transforms, download=True)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(train_data.data[i]))
plt.show()