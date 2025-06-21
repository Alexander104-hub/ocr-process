import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf


WIDTH = 732
HEIGHT = 32

csv_file = "handwritting/dataset/data.csv"
df = pd.read_csv(csv_file, nrows=35000)[-20:]

print(df.info())


def crop_white_borders(binary_img):
    top = 0
    while np.all(binary_img[top] == 255):
        top += 1
    bottom = binary_img.shape[0] - 1
    while np.all(binary_img[bottom] == 255):
        bottom -= 1
    left = 0
    while np.all(binary_img[:, left] == 255):
        left += 1
    right = binary_img.shape[1] - 1
    while np.all(binary_img[:, right] == 255):
        right -= 1
    return binary_img[top : bottom + 1, left : right + 1]


def resize_img(img, width_to, height_to):
    img_h, img_w = img.shape[:2]
    scale_w = width_to / img_w
    scale_h = height_to / img_h
    scale = min(scale_w, scale_h)
    new_w = int(img_w * scale + 0.5)
    new_h = int(img_h * scale + 0.5)
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    border_w = width_to - new_w
    border_h = height_to - new_h
    if border_w >= border_h:
        border_right = np.full((new_h, border_w), 255, dtype=np.uint8)
        img = np.hstack((img, border_right))
    else:
        border_top = np.full((border_h, new_w), 255, dtype=np.uint8)
        img = np.vstack((border_top, img))

    return img


def texts_to_nums(texts):
    alphabet = "".join(sorted(set("".join(texts))))
    nums = np.ones([len(texts), max(map(len, texts))], dtype="int64") * len(alphabet)
    for i, text in enumerate(texts):
        nums[i, : len(text)] = [alphabet.index(ch) for ch in text]
    return nums, alphabet


X, y = [], []

for index, row in df.iterrows():
    img_path = os.path.join(
        "handwritting",
        "dataset",
        "data",
        row["path"].split("/")[0],
        row["path"].split("/")[1],
    )
    img = cv2.imread(img_path, 0)
    img = crop_white_borders(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1])
    img = resize_img(img, WIDTH, HEIGHT)
    X.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).astype("bool"))
    y.append(row["label"].lower())

X = np.array(X)
y, alphabet = texts_to_nums(y)


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=43)
train_X, test_X, train_y, test_y = train_test_split(
    train_X, train_y, test_size=0.25, random_state=43
)


# fig, axes = plt.subplots(figsize=(20, 15), ncols=2, nrows=10)


class AugmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=64, apply_augmentation=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.apply_augmentation = apply_augmentation
        self.indexes = np.arange(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Create batch
        batch_X = np.array([self.X[i] for i in batch_indexes])
        batch_y = np.array([self.y[i] for i in batch_indexes])

        # Apply augmentation if enabled
        if self.apply_augmentation:
            batch_X = np.array([self.apply_weak_augmentation(img) for img in batch_X])

        return batch_X, batch_y

    def apply_weak_augmentation(self, img):
        """Apply weak augmentation to the image"""
        augmented = img.copy()

        # Only apply augmentation with 50% probability
        if np.random.random() < 1:
            return augmented

        # Randomly apply different augmentations - focus on noise and very gentle distortions
        aug_type = np.random.choice(
            ["minimal_rotation", "noise", "brightness", "elastic"], p=[0, 0, 0, 1]
        )

        if aug_type == "minimal_rotation":
            # Extremely small rotation (-1 to 1 degrees) to avoid cutting off text
            angle = np.random.uniform(-1, 1)
            h, w = augmented.shape[:2] if len(augmented.shape) > 2 else augmented.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Use a larger output size to ensure no text is cut off
            augmented = cv2.warpAffine(
                augmented.astype(np.uint8),
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            augmented = augmented.astype("bool") if img.dtype == "bool" else augmented

        elif aug_type == "noise":
            # Add a small amount of noise
            noise_level = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% noise
            if np.random.random() < 0.5:
                # Salt noise (white pixels)
                salt_mask = np.random.random(augmented.shape) < noise_level
                augmented = augmented.copy()
                augmented[salt_mask] = True if img.dtype == "bool" else 255
            else:
                # Pepper noise (black pixels)
                pepper_mask = np.random.random(augmented.shape) < noise_level
                augmented = augmented.copy()
                augmented[pepper_mask] = False if img.dtype == "bool" else 0

        elif aug_type == "brightness":
            # Slightly adjust brightness without moving pixels
            if np.random.random() < 0.5:
                # Darken slightly
                if img.dtype == "bool":
                    # For boolean images, randomly flip a very small percentage of white pixels to black
                    mask = np.logical_and(
                        augmented, np.random.random(augmented.shape) < 0.03
                    )
                    augmented = augmented.copy()
                    augmented[mask] = False
                else:
                    # For grayscale images, darken slightly
                    augmented = np.clip(
                        augmented.astype(np.int16) - np.random.randint(5, 15), 0, 255
                    ).astype(np.uint8)
            else:
                # Lighten slightly
                if img.dtype == "bool":
                    # For boolean images, randomly flip a very small percentage of black pixels to white
                    mask = np.logical_and(
                        ~augmented, np.random.random(augmented.shape) < 0.03
                    )
                    augmented = augmented.copy()
                    augmented[mask] = True
                else:
                    # For grayscale images, lighten slightly
                    augmented = np.clip(
                        augmented.astype(np.int16) + np.random.randint(5, 15), 0, 255
                    ).astype(np.uint8)

        elif aug_type == "elastic":
            # Very subtle elastic deformation for handwriting variation
            if img.dtype == "bool":
                # Convert to uint8 for processing
                temp_img = augmented.astype(np.uint8) * 255
            else:
                temp_img = augmented.copy()

            # Create displacement fields
            h, w = temp_img.shape[:2]
            # Use very small displacement to avoid losing text
            alpha = np.random.uniform(2, 5)
            sigma = np.random.uniform(3, 4)

            # Create random displacement fields
            dx = np.random.rand(h, w) * 2 - 1
            dy = np.random.rand(h, w) * 2 - 1

            # Smooth displacement fields
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)

            # Normalize displacement fields
            dx = alpha * dx / np.max(np.abs(dx))
            dy = alpha * dy / np.max(np.abs(dy))

            # Create meshgrid
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Displace meshgrid
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)

            # Remap image
            distorted = cv2.remap(
                temp_img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            # Convert back to original dtype
            if img.dtype == "bool":
                augmented = distorted > 127
            else:
                augmented = distorted

        return augmented

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        np.random.shuffle(self.indexes)


# Create data generators
train_generator = AugmentationDataGenerator(
    train_X, train_y, batch_size=64, apply_augmentation=True
)
val_generator = AugmentationDataGenerator(
    val_X, val_y, batch_size=64, apply_augmentation=False
)


import matplotlib.pyplot as plt

x_batch, y_batch = next(iter(train_generator))


n_images_to_show = 8


fig, axes = plt.subplots(1, n_images_to_show, figsize=(15, 4))


mng = plt.get_current_fig_manager()
try:
    mng.full_screen_toggle()
except AttributeError:
    try:
        mng.window.state("zoomed")
    except:
        pass

for i in range(n_images_to_show):
    img = x_batch[i]
    if img.shape[-1] == 1:
        img = img.squeeze(-1)

    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")

fig.suptitle("Augmented Images from Generator", fontsize=20)
plt.tight_layout()
plt.show()
