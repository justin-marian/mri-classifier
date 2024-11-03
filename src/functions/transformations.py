from torchvision import transforms


class Transformations:
    """ Create and customize transformations for image data preprocessing."""

    def __init__(self, resize=(224, 224), mean=None, std=None):
        """ Transformations: resizing, normalization, ... """
        self.resize = resize
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]

    def get_basic_transform(self):
        """ Transformation pipeline including resizing, tensor conversion, and normalization. """
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_augmentation_transform(self):
        """ Augmented transformation pipeline with random flips and rotation. """
        return transforms.Compose([
            transforms.Resize(self.resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
