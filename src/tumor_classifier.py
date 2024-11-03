""" tumor_classifier.py """

from torch.utils.data import DataLoader
from config import DIR_TRAINING, DIR_TESTING, init_conf
from utils.prepdata import PrepData
from utils.dataset import BrainTumorDataset
from funcs.transformer import Transformer
from models.plots import Plotter


if __name__ == "__main__":
    print(f"Preprocessing data for Training, Validating, Testing Samples...", flush=True)
    data_prep = PrepData(config=init_conf, train_dir=DIR_TRAINING)

    print(f"[TRAINING DIRECTORY]: Resizing images in training and validation sets...", flush=True)
    data_prep.resize_images()
    print(f"[TRAINING DIRECTORY]:", flush=True)
    data_prep.summary()

    train_dataset = BrainTumorDataset(
        image_paths=data_prep.train_images, 
        labels=data_prep.train_labels,
        transform=Transformer().get_basic_transform()
    )

    valid_dataset = BrainTumorDataset(
        image_paths=data_prep.valid_images,
        labels=data_prep.valid_labels,
        transform=Transformer().get_basic_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"\nTotal number of training images: {len(train_dataset)}", flush=True)
    print(f"Total number of validation images: {len(valid_dataset)}", flush=True)
    print(f"----------------------------------------\n", flush=True)

    # Prepare testing images
    test_prep = PrepData(config=init_conf, test_dir=DIR_TESTING)
    print(f"[TESTING DIRECTORY]:", flush=True)
    test_prep.summary()

    test_dataset = BrainTumorDataset(root_dir=DIR_TESTING, transform=Transformer().get_basic_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"\nTotal number of testing images: {len(test_dataset)}", flush=True)
    print(f"----------------------------------------\n", flush=True)

    plotter = Plotter(xlabel="Tumor Categories", ylabel="Number of Images", save_dir="../images")

    plotter.plot_combined_bar_charts(
        data_prep.train_class_counts,
        data_prep.valid_class_counts,
        test_prep.test_class_counts,
        save_name="before"
    )

    plotter.plot_combined_histograms(
        data_prep.train_class_counts,
        data_prep.valid_class_counts,
        test_prep.test_class_counts,
        save_name="before"
    )
