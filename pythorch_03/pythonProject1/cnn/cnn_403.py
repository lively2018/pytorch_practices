import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop

transforms = Compose(
    [
        T.ToPILImage(),
        RandomCrop((32, 32), padding=4),
        RandomHorizontalFlip(p=0.5),
    ]
)

train_data = CIFAR10(root='./data', train=True, transform=transforms, download=True)

test_data = CIFAR10(root='./data', train=False, transform=transforms, download=True)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(train_data.data[i]))
plt.show()