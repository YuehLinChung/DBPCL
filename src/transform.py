from torchvision import transforms

class ContrastiveTransformations:
    def __init__(self, base_transforms=None, n_views=2):
        if base_transforms is None:
            base_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(size=32),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]