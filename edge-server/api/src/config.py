"""
Edge server config
"""
# will be env var
import os

ENABLE_GPU = not bool(os.environ.get("ENABLE_GPU") == "FALSE")

TRAINING_IMAGE = os.environ.get("TRAINING_IMAGE")

HOST_DATASET_DIR = os.environ.get("HOST_DATASET_DIR")
