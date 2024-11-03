""" utils/prepdata.py """

import os
import random
from PIL import Image
from config.config import Config
from funcs.transformer import Transformer


class PrepData:

    def __init__(self, config: Config, train_dir=None, test_dir=None, target_size=(256, 256), train_ratio=0.8):
        """ Initialize with the training and/or testing directory, gather image information, and split training data if applicable. """
        self.config = config
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.transformer = Transformer(resize=self.target_size)

        # Gather images from Training folder and split it into training and validation sets
        if self.train_dir:
            self.train_class_map, self.train_images, self.train_labels = self._gather_images(self.train_dir)
            self.num_train_total = len(self.train_labels)
            self.train_images, self.train_labels, self.valid_images, self.valid_labels = self.split_train_val()

            # Calculate counts for training and validation sets
            self.train_class_counts = self._count_images_per_class(self.train_labels, self.train_class_map)
            self.valid_class_counts = self._count_images_per_class(self.valid_labels, self.train_class_map)

        # If a testing directory is provided, gather images for testing set without splitting
        if self.test_dir:
            self.test_class_map, self.test_images, self.test_labels = self._gather_images(self.test_dir)
            self.num_test = len(self.test_labels)
            self.test_class_counts = self._count_images_per_class(self.test_labels, self.test_class_map)

        # Get image dimensions from a sample image
        self.image_width, self.image_height = self._get_image_dimensions()

    def _get_class_map(self, directory):
        """ Create a mapping from original class names to formatted class names. """
        class_map = {}
        for class_name in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, class_name)) and class_name in self.config.CATEGORIES:
                formatted_name = class_name.replace('_', ' ').title()
                class_map[class_name] = formatted_name
        return class_map

    def _gather_images(self, directory):
        """ Gather image paths and corresponding class indices from a given directory. """
        class_map = self._get_class_map(directory)
        image_files = []
        labels = []

        for i, (orig_name, _) in enumerate(class_map.items()):
            class_dir = os.path.join(directory, orig_name)
            class_images = [os.path.join(class_dir, x) for x in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, x))]
            image_files.extend(class_images)
            labels.extend([i] * len(class_images))
        return class_map, image_files, labels

    def _count_images_per_class(self, labels, class_map):
        """ Count images per class based on labels list. """
        counts = {name: 0 for name in class_map.values()}
        for label in labels:
            class_name = list(class_map.values())[label]
            counts[class_name] += 1
        return counts

    def _get_image_dimensions(self):
        """ Get dimensions of the first image to establish consistency. """
        sample_image_path = (
            self.train_images[0] if hasattr(self, "train_images") else
            self.test_images[0] if hasattr(self, "test_images") else None
        )
        if sample_image_path:
            with Image.open(sample_image_path) as img:
                return img.size
        return None, None

    def resize_images(self):
        """ Resize all images in the training and validation sets to the target size. """
        if hasattr(self, "train_images") and hasattr(self, "valid_images"): # only to train images
            self.transformer.resize_image_files(self.train_images + self.valid_images, self.target_size)

    def split_train_val(self):
        """ Split the training data into training and validation sets based on the train_ratio. """
        random.seed(42)
        combined = list(zip(self.train_images, self.train_labels))
        random.shuffle(combined)
        
        split_index = int(len(combined) * self.train_ratio)
        train_data = combined[:split_index]
        valid_data = combined[split_index:]

        train_images, train_labels = zip(*train_data)
        valid_images, valid_labels = zip(*valid_data)

        self.num_train = len(train_images)
        self.num_val = len(valid_images)

        return list(train_images), list(train_labels), list(valid_images), list(valid_labels)

    def summary(self):
        """ Print a summary of the dataset information. """
        print(f"Image dimensions: {self.target_size[0]} x {self.target_size[1]}")

        if hasattr(self, "train_class_map"):
            print("Label names (Training):", list(self.train_class_map.values()))
            print(f"Total training set image count (including validation): {self.num_train_total}")
            print(f"Training set image count: {self.num_train}")
            print(f"Validation set image count: {self.num_val}")
            print("\nTraining class counts:")
            for class_name, count in self.train_class_counts.items():
                print(f" - {class_name}: {count}")
            print("\nValidation class counts:")
            for class_name, count in self.valid_class_counts.items():
                print(f" - {class_name}: {count}")

        if hasattr(self, "test_class_map"):
            print("Label names (Testing):", list(self.test_class_map.values()))
            print(f"Testing set image count: {self.num_test}")
            print("\nTesting class counts:")
            for class_name, count in self.test_class_counts.items():
                print(f" - {class_name}: {count}")
