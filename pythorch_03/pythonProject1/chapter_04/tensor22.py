import numpy as np
from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa

class IaaTransform:
    def __init__(self):
        self.seq = iaa.Sequential(
            [
                iaa.SaltAndPepper(p=(0.03, 0.07)),
                iaa.Rain(speed=(0.3, 0.7))
            ]
        )

    def __call__(self, images):
        images = np.array(images)
        augmented = self.seq.augment_images(images)
        return Image.fromarray(augmented)

transform = transforms.Compose(
    [
        IaaTransform.Resize((512, 512))
    ]
)
