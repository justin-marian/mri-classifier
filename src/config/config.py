""" config/config.py """

import os


class Config:

    def __init__(self, use_env_dir=False):
        # All possible categories of brain tumors
        self.CATEGORIES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

        if use_env_dir:
            # Use the directory specified in DATA_DIRECTORY environment variable
            self.DIR_ROOT = os.environ.get("DATA_DIRECTORY", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
        else:
            # Default to project root directory if use_env_dir is False
            self.DIR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        
        # Paths for Training, Testing, meta, and other folders within project root
        self.DIR_TRAINING = os.path.join(self.DIR_ROOT, "data", "Training")
        self.DIR_TESTING = os.path.join(self.DIR_ROOT, "data", "Testing")
        self.DIR_ARCHIVE = os.path.join(self.DIR_ROOT, "archive")
        self.DIR_META = os.path.join(self.DIR_ROOT, "meta")
        
        os.makedirs(self.DIR_TRAINING, exist_ok=True)
        os.makedirs(self.DIR_TESTING, exist_ok=True)
        os.makedirs(self.DIR_META, exist_ok=True)
        os.makedirs(self.DIR_ARCHIVE, exist_ok=True)

        self._print_directories()

    def _print_directories(self):
        """ Print the configured directories for debugging purposes. """
        print(f"Project Root Directory: {self.DIR_ROOT}", flush=True)
        print(f"Training Directory: {self.DIR_TRAINING}", flush=True)
        print(f"Testing Directory: {self.DIR_TESTING}", flush=True)
        print(f"Meta Directory: {self.DIR_META}", flush=True)
        print(f"Archive Directory: {self.DIR_ARCHIVE}", flush=True)
