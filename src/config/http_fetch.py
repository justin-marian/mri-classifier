""" config/http_fetch.py """

import os
import stat
import time
import shutil
import logging
from monai.apps import download_and_extract


class HttpFetch:
    """ Download and extract a dataset zip file from a URL, with logging support. """

    def __init__(self, url, sha1, archive_path="archive", data_path="data", log_dir="meta"):
        self.url = url
        self.sha1 = sha1
        self.data_path = os.path.abspath(data_path)
        self.archive_path = os.path.abspath(archive_path)
        self.log_dir = os.path.abspath(log_dir)
        self.archive_name = "dataset.zip"
        self.logger = logging.getLogger("HttpFetch")
        self.setup_logger()

    def setup_logger(self):
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
        zip_path = os.path.join(self.archive_path, self.archive_name)

        self.logger.info("Starting download and extraction using MONAI with SHA-1 verification...")
        download_and_extract(
            url=self.url,
            filepath=zip_path,
            output_dir=self.archive_path,
            hash_val=self.sha1,
            hash_type="sha1",
        )

        self.logger.info(f"Downloaded and extracted dataset to {self.archive_path}")
        extracted_root = os.path.join(self.archive_path, os.listdir(self.archive_path)[0])

        for folder_name in ["Training", "Testing"]:
            src_folder = os.path.join(extracted_root, folder_name)
            dest_folder = os.path.join(self.data_path, folder_name)

            if os.path.exists(dest_folder):
                self.logger.info(f"Removing existing {folder_name} folder at destination to prevent conflicts.")
                shutil.rmtree(dest_folder, onexc=self.remove_readonly)

            if os.path.exists(src_folder):
                shutil.move(src_folder, dest_folder)
                self.logger.info(f"Moved {folder_name} folder to {self.data_path}")
            else:
                self.logger.error(f"{folder_name} folder not found in the extracted dataset.")
                raise FileNotFoundError(f"{folder_name} folder not found in the extracted dataset.")

        shutil.rmtree(extracted_root, onexc=self.remove_readonly)
        self.logger.info(f"Removed original extraction directory: {extracted_root}")
        os.remove(zip_path)
        self.logger.info("Download, extraction, and reorganization complete.\n")

    def remove_readonly(self, func, path):
        """ Retry removing the file after clearing readonly and waiting briefly. """
        os.chmod(path, stat.S_IWRITE)
        time.sleep(0.5)
        func(path)
