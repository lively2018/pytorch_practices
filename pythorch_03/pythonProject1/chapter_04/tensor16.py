from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
)

transform2 = transforms.Compose(
    [
        transforms.RandomRotation(degrees=30, expand=False, center=None),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ]
)

transform3 = transforms.Compose(
    [
        transforms.RandomCrop(size=(512,512)),
        transforms.Pad(padding=50, fill=(127, 127, 255), padding_mode='constant')
    ]
)

transform4 = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=15, translate=(0.2, 0.2),
            scale=(0.8, 1.2), shear=15
        )
    ]
)

transform5 = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        ),
        transforms.ToPILImage()
    ]
)
transform6 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, value=0),
        transforms.RandomErasing(p=1.0, value="random"),
        transforms.ToPILImage()
    ]
)
image = Image.open("/home/kssong/dataset/pytorch_practice/datasets/images/cat.jpg")
transform_image = transform(image)
print(transform_image.shape)