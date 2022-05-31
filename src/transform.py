from torchvision import transforms

class ContrastiveTransformations:
    def __init__(self, base_transforms=None, n_views=2, crop_size=32, gaussian_blur=True, jitter_strength=1):
        self.n_views = n_views
        self.crop_size = crop_size
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        
        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )
        data_transforms = [
            transforms.RandomResizedCrop(size=self.crop_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if self.gaussian_blur:
            kernel_size = int(0.1 * self.crop_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            data_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))
        
        data_transforms = transforms.Compose(data_transforms)
        self.final_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train_transform = transforms.Compose([data_transforms, self.final_transform])
        
    def __call__(self, x):
        return [self.train_transform(x) for i in range(self.n_views)]
    
class MoCov1Transformation:
    def __init__(self, n_views=2, crop_size=32):
        self.n_views = n_views
        self.crop_size = crop_size
        augmentation = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        self.transform = transforms.Compose(augmentation)
    
    def __call__(self, x):
        return [self.transform(x) for i in range(self.n_views)]

class MoCov2Transformation:
    def __init__(self, n_views=2, crop_size=32):
        self.n_views = n_views
        self.crop_size = crop_size
        kernel_size = int(0.1 * self.crop_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        augmentation = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size,sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        self.transform = transforms.Compose(augmentation)
    
    def __call__(self, x):
        return [self.transform(x) for i in range(self.n_views)]
    
class WeekTransformation:
    def __init__(self, n_views=2, crop_size=32):
        self.n_views = n_views
        
        self.transform = [transforms.RandomResizedCrop(size=crop_size,scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ]

        self.transform += [transforms.Normalize((0.5,), (0.5,))]
        
        self.transform = transforms.Compose(self.transform)
    
    def __call__(self, x):
        return [self.transform(x) for i in range(self.n_views)]
