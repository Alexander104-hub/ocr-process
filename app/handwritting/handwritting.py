import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode, get_value
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU


class HandwritingRecognitionModel:
    def __init__(self, model_path="model/my_model.h5"):
        """
        Initialize the handwriting recognition model

        Args:
            model_path (str): Path to the trained model
        """

        class CERMetric(tf.keras.metrics.Metric):
            def __init__(self, name="CER_metric", **kwargs):
                super(CERMetric, self).__init__(name=name, **kwargs)
                self.cer_accumulator = self.add_weight(
                    name="total_cer", initializer="zeros"
                )
                self.counter = self.add_weight(name="cer_count", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None):
                input_shape = tf.keras.backend.shape(y_pred)
                input_length = tf.ones(shape=input_shape[0]) * tf.cast(
                    input_shape[1], "float32"
                )
                decode, log = tf.keras.backend.ctc_decode(
                    y_pred, input_length, greedy=True
                )
                decode = tf.keras.backend.ctc_label_dense_to_sparse(
                    decode[0], tf.cast(input_length, "int32")
                )
                y_true_sparse = tf.keras.backend.ctc_label_dense_to_sparse(
                    y_true, tf.cast(input_length, "int32")
                )
                y_true_sparse = tf.sparse.retain(
                    y_true_sparse,
                    tf.not_equal(
                        y_true_sparse.values, tf.math.reduce_max(y_true_sparse.values)
                    ),
                )
                decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
                distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
                self.cer_accumulator.assign_add(tf.reduce_sum(distance))
                self.counter.assign_add(tf.cast(tf.shape(y_true)[0], "float32"))

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
            loss = tf.keras.backend.ctc_batch_cost(
                y_true, y_pred, input_length, label_length
            )
            return loss

        # Define the alphabet based on what was used during training
        self.alphabet = " (),-.:;абвгдежзийклмнопрстуфхцчшщъыьэюяё"

        # Load the model with custom objects
        self.model = load_model(
            "model/my_model.h5",
            custom_objects={
                "LeakyReLU": LeakyReLU,
                "CTCLoss": CTCLoss,
                "CERMetric": CERMetric,
            },
        )

        # Model parameters
        self.WIDTH = 732
        self.HEIGHT = 32

    def preprocess_image(self, img):
        """
        Preprocess an image for the model

        Args:
            img (numpy.ndarray): Input image (grayscale)

        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Crop white borders
        binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        top = 0
        while top < binary_img.shape[0] and np.all(binary_img[top] == 255):
            top += 1
        bottom = binary_img.shape[0] - 1
        while bottom >= 0 and np.all(binary_img[bottom] == 255):
            bottom -= 1
        left = 0
        while left < binary_img.shape[1] and np.all(binary_img[:, left] == 255):
            left += 1
        right = binary_img.shape[1] - 1
        while right >= 0 and np.all(binary_img[:, right] == 255):
            right -= 1

        # Safety check - if the image is entirely white
        if top >= bottom or left >= right:
            return np.full((self.HEIGHT, self.WIDTH), 255, dtype=np.uint8)

        cropped_img = binary_img[top : bottom + 1, left : right + 1]

        # Resize image
        img_h, img_w = cropped_img.shape[:2]
        scale_w = self.WIDTH / img_w
        scale_h = self.HEIGHT / img_h
        scale = min(scale_w, scale_h)
        new_w = int(img_w * scale + 0.5)
        new_h = int(img_h * scale + 0.5)

        resized_img = cv2.resize(cropped_img, (new_w, new_h))
        resized_img = cv2.adaptiveThreshold(
            resized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Add padding to reach target dimensions
        border_w = self.WIDTH - new_w
        border_h = self.HEIGHT - new_h

        if border_w >= border_h:
            border_right = np.full((new_h, border_w), 255, dtype=np.uint8)
            processed_img = np.hstack((resized_img, border_right))
        else:
            border_top = np.full((border_h, new_w), 255, dtype=np.uint8)
            processed_img = np.vstack((border_top, resized_img))

        # Rotate the image 90 degrees clockwise (as done in training)
        processed_img = cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE)

        return processed_img

    def predict(self, img):
        """
        Recognize text in the provided image

        Args:
            img (numpy.ndarray): Input image (grayscale)

        Returns:
            str: Recognized text
        """
        # Preprocess the image
        processed_img = self.preprocess_image(img)

        # Convert to boolean array as used in training
        processed_img = processed_img.astype("bool")

        # Add batch and channel dimensions
        input_img = np.reshape(processed_img, (1, self.WIDTH, self.HEIGHT, 1))

        # Get model prediction
        prediction = self.model.predict(input_img, verbose=0)

        # Decode the prediction
        decoded_text = self.decode_text(prediction)

        return decoded_text[0] if decoded_text else ""

    def decode_text(self, prediction):
        """
        Decode model prediction to text

        Args:
            prediction (numpy.ndarray): Model prediction

        Returns:
            list: List of decoded texts
        """
        # Decode using CTC
        input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
        decoded = get_value(
            ctc_decode(prediction, input_length=input_length, greedy=True)[0][0]
        )

        # Convert indices to characters
        texts = []
        for values in decoded:
            text = "".join(
                [self.alphabet[i] for i in values if i >= 0 and i < len(self.alphabet)]
            )
            texts.append(text)

        return texts


class ImagePolygonProcessor:
    def __init__(self, tmp_dir="tmp"):
        """
        Initialize the image polygon processor

        Args:
            tmp_dir (str): Directory for temporary output images
        """
        self.tmp_dir = tmp_dir
        # Create or clear tmp directory
        self.clear_tmp_directory()

    def clear_tmp_directory(self):
        """
        Clear the temporary directory
        """
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def read_polygon_coordinates(self, gt_file):
        """
        Read polygon coordinates from a ground truth file

        Args:
            gt_file (str): Path to the ground truth file

        Returns:
            list: List of polygon coordinates
        """
        polygons = []
        with open(gt_file, "r") as f:
            for line in f:
                if line.strip():
                    # Parse coordinates
                    coords = list(map(int, line.strip().split(",")))
                    # Create array of points (x1,y1,x2,y2,x3,y3,x4,y4) -> [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    points = np.array(
                        [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
                    )
                    polygons.append(points)
        return polygons

    def extract_polygons(self, image_path, gt_file, recognize_text=False, model=None):
        """
        Extract text regions from an image based on polygon coordinates

        Args:
            image_path (str): Path to the image
            gt_file (str): Path to the ground truth file with polygon coordinates
            recognize_text (bool): Whether to recognize text in each polygon
            model (HandwritingRecognitionModel): Model for text recognition

        Returns:
            list: List of extracted images and recognized texts if requested
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Create a grayscale version for text recognition
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Read polygon coordinates
        polygons = self.read_polygon_coordinates(gt_file)

        results = []

        # Process each polygon
        for i, points in enumerate(polygons):
            # Ensure points are integer type and have the correct shape
            points = points.astype(np.int32)

            # Get bounding rectangle for the polygon
            x, y, w, h = cv2.boundingRect(points)

            # Crop the original image
            cropped_img = img[y : y + h, x : x + w].copy()
            cropped_gray = gray_img[y : y + h, x : x + w].copy()

            # Create a mask for the polygon with correct dimensions
            mask = np.zeros((h, w), dtype=np.uint8)

            # Shift polygon points to the cropped image coordinates
            shifted_points = points - np.array([x, y])

            # Fill the polygon in the mask
            cv2.fillPoly(mask, [shifted_points], 255)
            result_img = np.ones_like(cropped_img) * 255

            for row in range(h):
                for col in range(w):
                    if mask[row, col] == 255:
                        if row < cropped_img.shape[0] and col < cropped_img.shape[1]:
                            result_img[row, col] = cropped_img[row, col]

            result = {
                "image": result_img,
                "gray_image": cropped_gray,
                "polygon": points,
                "box": (x, y, w, h),
            }

            # Recognize text if requested
            if recognize_text and model:
                masked_gray = np.ones_like(cropped_gray) * 255
                for row in range(h):
                    for col in range(w):
                        if (
                            mask[row, col] == 255
                            and row < cropped_gray.shape[0]
                            and col < cropped_gray.shape[1]
                        ):
                            masked_gray[row, col] = cropped_gray[row, col]

                text = model.predict(masked_gray)
                result["text"] = text

            results.append(result)

        return results
