""" utils/dataset.py """

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from config import init_conf


class BrainTumorDataset(Dataset):
    """ Dataset for brain tumor MRI images, supporting both directory-based and list-based initialization. """

    def __init__(self, root_dir=None, image_paths=None, labels=None, transform=None):
        """ Initialize the dataset with either a root directory or lists of image paths and labels, with optional transform. """
        self.cache = {}
        self.transform = transform

        if root_dir:
            self.root_dir = Path(root_dir)
            self.image_paths, self.labels = self._load_paths_and_labels()
            self.label_map = {idx: category for idx, category in enumerate(init_conf.CATEGORIES)}
        elif image_paths and labels:
            self.image_paths = image_paths
            self.labels = labels
            self.label_map = {idx: category for idx, category in enumerate(init_conf.CATEGORIES)}
        else:
            raise ValueError("Either root_dir or image_paths and labels must be provided.")

    def _load_paths_and_labels(self):
        """ Load image paths and labels from the specified directory. """
        image_paths = []
        labels = []
        label_map = {category: idx for idx, category in enumerate(init_conf.CATEGORIES)}

        for category, label in label_map.items():
            category_path = self.root_dir / category
            if not category_path.exists() or not category_path.is_dir():
                continue

            for entry in category_path.glob("*"):
                if entry.is_file():
                    try:
                        with Image.open(entry) as img:
                            img.verify()
                        image_paths.append(entry)
                        labels.append(label)
                    except (IOError, SyntaxError):
                        print(f"File {entry} is not a valid image and will be skipped.", flush=True)

        return image_paths, labels

    def __len__(self):
        """ Return the total number of images in the dataset. """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Return the image and label name corresponding to the given index, with caching. """
        if idx in self.cache:
            return self.cache[idx]

        image_path = self.image_paths[idx]
        label_idx = self.labels[idx]
        label_name = init_conf.CATEGORIES[label_idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        self.cache[idx] = (image, label_name)
        return image, label_name
