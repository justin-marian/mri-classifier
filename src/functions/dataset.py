import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.config.config import CATEGORIES

class BrainTumorDataset(Dataset):
    """ Lazy loading dataset only when they are necessary for batches. """

    def __init__(self, root_dir, transform=None):
        """ Dataset for brain tumors based oon images from MRI. """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths, self.labels = self._load_paths_and_labels()

    def _load_paths_and_labels(self):
        """ Lazy loading path images and labels. """
        image_paths = []
        labels = []

        for label, category in enumerate(CATEGORIES):
            category_path = self.root_dir / category
            if not category_path.exists() or not category_path.is_dir():
                continue

            # List all image files in the category directory
            with os.scandir(category_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        image_paths.append(entry.path)
                        labels.append(label)

        return image_paths, labels

    def __len__(self):
        """ Total number of images in the dataset. """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Load and return image and label corresponding to the given index. """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
