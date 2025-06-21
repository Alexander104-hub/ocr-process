import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import get_value, ctc_decode
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Bidirectional, Dense, LSTM, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam



WIDTH = 732
HEIGHT = 32

csv_file = "handwritting/dataset/data.csv"
df = pd.read_csv(csv_file)

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
    return binary_img[top:bottom + 1, left:right + 1]


def resize_img(img, width_to, height_to):
    img_h, img_w = img.shape[:2]
    scale_w = width_to / img_w
    scale_h = height_to / img_h
    scale = min(scale_w, scale_h)
    new_w = int(img_w * scale + 0.5)
    new_h = int(img_h * scale + 0.5)
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
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
  alphabet = ''.join(sorted(set(''.join(texts))))
  nums = np.ones([len(texts), max(map(len, texts))], dtype='int64') * len(alphabet)
  for i, text in enumerate(texts):
    nums[i, :len(text)] = [alphabet.index(ch) for ch in text]
  return nums, alphabet

X, y = [], []

for index, row in df.iterrows():
    img_path = os.path.join("handwritting", "dataset", "data", 
                            row["path"].split('/')[0], row["path"].split('/')[1])
    img = cv2.imread(img_path, 0)
    try:
        img = crop_white_borders(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1])
    except Exception as e:
        print(e)
    img = resize_img(img, WIDTH, HEIGHT)
    if row['label'] and isinstance(row['label'], str):
        X.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).astype('bool'))
        y.append(row['label'].lower())
# ALPHABET::: " !%'()+,-./0123456789:;=?[]bcehioprstuxy«»абвгдежзийклмнопрстуфхцчшщъыьэюяё№"
X = np.array(X)
y, alphabet = texts_to_nums(y)
print(alphabet)

train_X, val_X, train_y, val_y  = train_test_split(X, y, test_size=0.2, random_state=43)
train_X, test_X, train_y, test_y  = train_test_split(train_X, train_y, test_size=0.25, random_state=43)


fig, axes = plt.subplots(figsize=(20, 15), ncols=2, nrows=10)

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
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
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
        if np.random.random() < 0.5:
            return augmented
        
        # Randomly apply different augmentations - focus on noise and very gentle distortions
        aug_type = np.random.choice(['minimal_rotation', 'noise', 'brightness', 'elastic'], 
                                   p=[0.3, 0.3, 0.2, 0.2])
        
        if aug_type == 'minimal_rotation':
            # Extremely small rotation (-1 to 1 degrees) to avoid cutting off text
            angle = np.random.uniform(-1, 1)
            h, w = augmented.shape[:2] if len(augmented.shape) > 2 else augmented.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Use a larger output size to ensure no text is cut off
            augmented = cv2.warpAffine(augmented.astype(np.uint8), M, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            augmented = augmented.astype('bool') if img.dtype == 'bool' else augmented
        
        elif aug_type == 'noise':
            # Add a small amount of noise
            noise_level = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5% noise
            if np.random.random() < 0.5:
                # Salt noise (white pixels)
                salt_mask = np.random.random(augmented.shape) < noise_level
                augmented = augmented.copy()
                augmented[salt_mask] = True if img.dtype == 'bool' else 255
            else:
                # Pepper noise (black pixels)
                pepper_mask = np.random.random(augmented.shape) < noise_level
                augmented = augmented.copy()
                augmented[pepper_mask] = False if img.dtype == 'bool' else 0
        
        elif aug_type == 'brightness':
            # Slightly adjust brightness without moving pixels
            if np.random.random() < 0.5:
                # Darken slightly
                if img.dtype == 'bool':
                    # For boolean images, randomly flip a very small percentage of white pixels to black
                    mask = np.logical_and(augmented, np.random.random(augmented.shape) < 0.03)
                    augmented = augmented.copy()
                    augmented[mask] = False
                else:
                    # For grayscale images, darken slightly
                    augmented = np.clip(augmented.astype(np.int16) - np.random.randint(5, 15), 0, 255).astype(np.uint8)
            else:
                # Lighten slightly
                if img.dtype == 'bool':
                    # For boolean images, randomly flip a very small percentage of black pixels to white
                    mask = np.logical_and(~augmented, np.random.random(augmented.shape) < 0.03)
                    augmented = augmented.copy()
                    augmented[mask] = True
                else:
                    # For grayscale images, lighten slightly
                    augmented = np.clip(augmented.astype(np.int16) + np.random.randint(5, 15), 0, 255).astype(np.uint8)
        
        elif aug_type == 'elastic':
            if img.dtype == 'bool':
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
            distorted = cv2.remap(temp_img, map_x, map_y, 
                                 interpolation=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=255)
            
            # Convert back to original dtype
            if img.dtype == 'bool':
                augmented = (distorted > 127)
            else:
                augmented = distorted
        
        return augmented
    
    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        np.random.shuffle(self.indexes)

# Create data generators
train_generator = AugmentationDataGenerator(train_X, train_y, batch_size=64, apply_augmentation=True)
val_generator = AugmentationDataGenerator(val_X, val_y, batch_size=64, apply_augmentation=False)


class CERMetric(tf.keras.metrics.Metric):
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        decode, log = K.ctc_decode(y_pred, input_length, greedy=True)
        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
        y_true_sparse = tf.sparse.retain(y_true_sparse, tf.not_equal(y_true_sparse.values, tf.math.reduce_max(y_true_sparse.values)))
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(tf.shape(y_true)[0], 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss



# Define the input shape
input_shape = (WIDTH, HEIGHT, 1)
inputs = Input(shape=input_shape)

# First block
x = Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01))(inputs)
x = MaxPooling2D((2, 2))(x)

# Second block
x = Conv2D(128, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01))(x)
x = MaxPooling2D((1, 2))(x)

# Third block
x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(x)
x = MaxPooling2D((2, 2))(x)
x = BatchNormalization()(x)

# Fourth block
x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(x)

# Fifth block
x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(x)
x = MaxPooling2D((2, 2))(x)

# Sixth block
x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01))(x)
x = MaxPooling2D((1, 2))(x)
x = BatchNormalization()(x)

# Reshape before LSTM
x = Reshape((91, 512))(x)

# Bidirectional LSTMs
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)

# Output layer
outputs = Dense(len(alphabet) + 1, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)



model.summary()
model.compile(optimizer=Nadam(learning_rate=0.001, clipnorm=1.0), loss  =CTCLoss, metrics=[CERMetric()])


model_checkpoint = ModelCheckpoint(filepath='model/my_model.h5', save_best_only=True)
# history = model.fit(train_X, train_y, validation_data=(val_X, val_y),
#                     epochs=100, batch_size=64,
#                     callbacks=[EarlyStopping(patience=15, restore_best_weights=True, monitor='val_CER_metric', mode='min'), 
#                                ReduceLROnPlateau(factor=0.5, min_lr=1e-5, patience=4, monitor='val_CER_metric', mode='min')], 
#                     verbose=1)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_CER_metric', mode='min'),
        ReduceLROnPlateau(factor=0.5, min_lr=1e-5, patience=4, monitor='val_CER_metric', mode='min'),
        model_checkpoint
    ],
    verbose=1
)


import json

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)


model.evaluate(test_X, test_y)


def num_to_label(num, alphabet):
    return ''.join(alphabet[ch] for ch in num if ch < len(alphabet))

def decode_text(nums, alphabet):
    values = get_value(ctc_decode(nums, input_length=np.ones(nums.shape[0])*nums.shape[1], greedy=True)[0][0])

    return [num_to_label(value[value >= 0], alphabet) for value in values]


predicts = model.predict(np.reshape(test_X, (-1, 732, 32, 1)))
predicts = decode_text(predicts, alphabet)
print(predicts)


