import matplotlib.pyplot as plt

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

# Load CIFAR-10 dataset
training_data = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(training_data.data[i])

plt.show()