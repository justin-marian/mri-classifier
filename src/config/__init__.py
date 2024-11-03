""" config/__init__.py """

import os
from .config import Config
from .cuda_info import CudaInfo
from .http_fetch import HttpFetch

# GitHub URL for Brain Tumor Classification (MRI) and SHA1 for zip dataset
URL = "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/archive/refs/heads/master.zip"
SHA1 = "6fbf6d0b328aa6db16b26c8a6b780f1e50052a70"

# Initialize configuration
init_conf = Config()
DIR_TRAINING = init_conf.DIR_TRAINING
DIR_TESTING = init_conf.DIR_TESTING
DIR_META = init_conf.DIR_META
DIR_ARCHIVE = init_conf.DIR_ARCHIVE
CATEGORIES = init_conf.CATEGORIES

# Initialize CUDA info and save it under DIR_META
cuda_info = CudaInfo(output_dir=DIR_META)
cuda_info.save_to_yaml()

# Set paths within the DIR_ROOT structure
ARCHIVE_PATH = DIR_ARCHIVE
DATA_PATH = os.path.join(init_conf.DIR_ROOT, "data")
LOG_PATH = DIR_META

# Initialize HttpFetch to download and extract dataset
fetcher = HttpFetch(URL, SHA1, ARCHIVE_PATH, DATA_PATH, LOG_PATH)
fetcher.fetch_extract()
