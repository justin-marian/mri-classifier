import os
import tempfile

# All possible categories of brain tumors
CATEGORIES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Environment variable (`DATA_DIRECTORY`) -> to use `meta` directory
#   for saving data results, otherwise, use a temporary director
DIR_DATA = os.environ.get("DATA_DIRECTORY", "meta")
DIR_ROOT = tempfile.mkdtemp() if not os.path.exists(DIR_DATA) else DIR_DATA
os.makedirs(DIR_ROOT, exist_ok=True)
print(f"Directory main for data: {DIR_ROOT}")

# Paths for Training and Testing sets
DIR_TRAINING = os.path.join(DIR_ROOT, "Training")
DIR_TESTING = os.path.join(DIR_ROOT, "Testing")
os.makedirs(DIR_TRAINING, exist_ok=True)
os.makedirs(DIR_TESTING, exist_ok=True)
print(f"Directory training: {DIR_TRAINING}")
print(f"Directory testing: {DIR_TESTING}")
