from torchvision import transforms

class ContrastiveTransformations:
    def __init__(self, base_transforms=None, n_views=2, crop_size=32, jitter_strength=1):
        self.crop_size = crop_size
        self.jitter_strength = jitter_strength
        if base_transforms is None:
            base_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(size=crop_size),
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        # ==============================================================
        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )
        resize_crop = transforms.RandomResizedCrop(size=crop_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.))
        random_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_transform = transforms.Compose([transforms.RandomApply([self.color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2)])
        
        kernel_size = int(0.1 * self.crop_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur_transform = transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5)
        final_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        
        self.entire_transforms = transforms.Compose([resize_crop, random_flip, color_transform, blur_transform, final_transform])
        # ==============================================================
        # self.base_transforms = base_transforms
        self.base_transforms = self.entire_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]