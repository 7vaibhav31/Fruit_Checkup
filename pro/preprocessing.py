"""
preprocessing.py
----------------
Prepares the fruit freshness dataset for training.
Matches exactly what was done in the Kaggle notebook.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DIR = "./fruits-fresh-and-rotten-for-classification/dataset/train"
TEST_DIR  = "./fruits-fresh-and-rotten-for-classification/dataset/test"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
CLASSES    = 6


def count_images(data_dir: str) -> None:
    """Print image counts per class."""
    for cls in os.listdir(data_dir):
        count = len(os.listdir(os.path.join(data_dir, cls)))
        print(f"{cls}: {count} images")


def get_data_generators():
    """
    Returns train_data and val_data generators.

    Notes
    -----
    * rescale=1/255  – pixel values normalised to [0, 1]
    * validation_split=0.2 on the TRAIN folder (80/20 split)
    * Test folder used as validation_data (no augmentation needed)
    * No additional augmentation because the dataset already contains
      augmented images.
    """
    datagen = ImageDataGenerator(
        rescale          = 1.0 / 255,
        validation_split = 0.2,
    )

    train_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        subset      = "training",
    )

    val_data = datagen.flow_from_directory(
        TEST_DIR,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "categorical",
        subset      = "validation",
    )

    return train_data, val_data


if __name__ == "__main__":
    print("=== Class Distribution ===")
    count_images(TRAIN_DIR)
    print()

    train_data, val_data = get_data_generators()

    print(f"\nTrain samples : {train_data.samples}")
    print(f"Val samples   : {val_data.samples}")
    print(f"Classes       : {train_data.class_indices}")
