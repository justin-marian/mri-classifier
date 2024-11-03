import os
import shutil
import requests
import logging
from zipfile import ZipFile
from typing import cast, AnyStr


class HttpFetcher:
    """Download and extract a dataset zip file from a URL, with logging support."""

    def __init__(self, url, archive_path="archive", archive_name="dataset.zip", data_path="data", log_dir="meta"):
        """Initialize the HttpFetcher with the URL, paths, and logging settings."""
        self.url = url
        self.data_path = data_path
        self.archive_path = archive_path
        self.archive_name = archive_name
        self.log_dir = log_dir
        self.logger = logging.getLogger("HttpFetcher")
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_logger()

    def setup_logger(self):
        """Set up the logger to log messages to both a file and the console."""
        log_path = os.path.join(self.log_dir, "http_fetch.log")
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def fetch_extract(self):
        """Download the zip file from the URL, extract it to the data path, and log each step."""
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.archive_path, exist_ok=True)
        zip_path = os.path.join(self.archive_path, self.archive_name)

        # Start downloading the zip file
        self.logger.info("Starting download...")
        response = requests.get(self.url, stream=True)
        with open(zip_path, "wb") as file:
            shutil.copyfileobj(response.raw, cast(AnyStr, file))
        self.logger.info(f"Downloaded dataset to {zip_path}")

        self.logger.info("Extracting dataset...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_path)
        self.logger.info(f"Dataset extracted to {self.data_path}")

        os.remove(zip_path)
        self.logger.info("Download and extraction complete. Temporary files removed.")
