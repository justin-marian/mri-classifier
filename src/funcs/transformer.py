""" funcs/transformers.py """

from PIL import Image
from torchvision import transforms


class Transformer:
    """ Create and customize transformations for image data preprocessing. """

    def __init__(self, resize=(256, 256), mean=None, std=None):
        """ Transformer: resizing, normalization, ... """
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

    def resize_image_files(self, image_files, target_size):
        """ Resize a list of images to the target size and save them. """
        for image_path in image_files:
            with Image.open(image_path) as img:
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(image_path)
        print(f"All images resized to {target_size}\n", flush=True)
