import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import json
import os
import cv2
import pandas as pd
from collections import Counter
import editdistance
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU


class CERMetric(tf.keras.metrics.Metric):
    def __init__(self, name="CER_metric", **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], "float32")
        decode, log = K.ctc_decode(y_pred, input_length, greedy=True)
        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, "int32"))
        y_true_sparse = K.ctc_label_dense_to_sparse(
            y_true, K.cast(input_length, "int32")
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
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# Load training history
with open("training_history.json", "r") as f:
    history = json.load(f)

# Function to calculate CER for individual samples
def calculate_cer(true_text, pred_text):
    """Calculate Character Error Rate between two strings"""
    distance = editdistance.eval(true_text, pred_text)
    return distance / len(true_text) if len(true_text) > 0 else 0


# Load the model
model = keras.models.load_model(
    "model/my_model.h5",
    custom_objects={"LeakyReLU": LeakyReLU, "CTCLoss": CTCLoss, "CERMetric": CERMetric},
)


# Define how to decode predictions from the model output
def ctc_decode(y_pred, input_length, greedy=True):
    """CTC decode function"""
    input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
    return K.ctc_decode(y_pred, input_length, greedy=greedy)


def get_value(tensor):
    """Convert tensor to numpy array"""
    return tensor.numpy()


def num_to_label(num, alphabet):
    """Convert numerical indices to text labels"""
    return "".join(alphabet[ch] for ch in num if ch < len(alphabet))


def decode_text(nums, alphabet):
    """Decode CTC outputs to text"""
    values = get_value(
        ctc_decode(
            nums, input_length=np.ones(nums.shape[0]) * nums.shape[1], greedy=True
        )[0][0]
    )
    return [num_to_label(value[value >= 0], alphabet) for value in values]


def load_test_data():
    print("Loading test data...")

    csv_file = "handwritting/dataset/data.csv"
    df = pd.read_csv(csv_file)

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
        alphabet = " (),-.:;абвгдежзийклмнопрстуфхцчшщъыьэюяё"
        nums = np.ones([len(texts), max(map(len, texts))], dtype="int64") * len(
            alphabet
        )
        for i, text in enumerate(texts):
            nums[i, : len(text)] = [alphabet.index(ch) for ch in text]
        return nums, alphabet

    from sklearn.model_selection import train_test_split

    WIDTH, HEIGHT = 732, 32
    X, y_texts = [], []

    for index, row in df.head(10200).iterrows():
        img_path = os.path.join(
            "handwritting",
            "dataset",
            "data",
            row["path"].split("/")[0],
            row["path"].split("/")[1],
        )
        try:
            img = cv2.imread(img_path, 0)
            img = crop_white_borders(cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1])
            img = resize_img(img, WIDTH, HEIGHT)
            X.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).astype("bool"))
            y_texts.append(row["label"].lower())
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    X = np.array(X)
    y, alphabet = texts_to_nums(y_texts)

    _, test_X, _, test_y = train_test_split(X, y, test_size=0.2, random_state=43)

    return test_X, test_y, y_texts, alphabet


try:
    test_X, test_y, original_texts, alphabet = load_test_data()
    print(f"Loaded {len(test_X)} test samples")
    print(f"Alphabet: {alphabet}")
except Exception as e:
    print(f"Error loading test data: {e}")


print("Getting model predictions...")
try:
    predicts_raw = model.predict(np.reshape(test_X, (-1, 732, 32, 1)))
    predicted_texts = decode_text(predicts_raw, alphabet)
    print(f"Made predictions for {len(predicted_texts)} samples")
except Exception as e:
    print(f"Error making predictions: {e}")


true_texts = []
for y in test_y:
    true_text = "".join([alphabet[idx] for idx in y if idx < len(alphabet)])
    true_texts.append(true_text)

print("Calculating CER metrics...")
cers = []
for true, pred in zip(true_texts, predicted_texts):
    cer = calculate_cer(true, pred)
    cers.append(cer)

print("Creating confusion matrix...")
all_true_chars = []
all_pred_chars = []
confusion_dict = {}

for true, pred in zip(true_texts, predicted_texts):
    # Collect character-level data for confusion matrix
    for t_char, p_char in zip(true, pred + " " * (len(true) - len(pred))):
        if len(p_char.strip()) == 0:
            p_char = "_"  # Use underscore for missing predictions
        all_true_chars.append(t_char)
        all_pred_chars.append(p_char)

        # Track character confusion pairs
        key = (t_char, p_char)
        if key not in confusion_dict:
            confusion_dict[key] = 0
        confusion_dict[key] += 1

# Create a set of all unique characters observed
all_chars = sorted(set(all_true_chars + all_pred_chars))

# Create a mapping from characters to indices for confusion matrix
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}

# Initialize confusion matrix
conf_matrix = np.zeros((len(all_chars), len(all_chars)))

# Fill confusion matrix
for true_char, pred_char in zip(all_true_chars, all_pred_chars):
    conf_matrix[char_to_idx[true_char], char_to_idx[pred_char]] += 1

# Normalize by rows to get percentages
row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_norm = np.zeros_like(conf_matrix, dtype=float)

# Track which characters are most often confused
confusion_counts = []
for i, true_char in enumerate(all_chars):
    for j, pred_char in enumerate(all_chars):
        if i != j and conf_matrix[i, j] > 0:
            confusion_counts.append(
                (
                    true_char,
                    pred_char,
                    conf_matrix[i, j],
                    conf_matrix[i, j] / row_sums[i][0] * 100,
                )
            )

# Sort by count in descending order
confusion_counts.sort(key=lambda x: x[2], reverse=True)

# Track character-specific errors
char_errors = {}
for true_char in set(all_true_chars):
    idx = char_to_idx[true_char]
    total = row_sums[idx][0]
    correct = conf_matrix[idx, idx]
    error_rate = 1 - (correct / total) if total > 0 else 0
    char_errors[true_char] = error_rate


# Create visualizations

# Set default figure size and style
plt.rcParams["figure.figsize"] = (12, 8)
sns.set_style("whitegrid")

# 1. Plot training history - Loss
plt.figure()
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot training history - CER
plt.figure()
plt.plot(history["CER_metric"], label="Training CER")
plt.plot(history["val_CER_metric"], label="Validation CER")
plt.title("Character Error Rate During Training")
plt.xlabel("Epoch")
plt.ylabel("CER")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Plot CER distribution for test set
plt.figure()
plt.hist(cers, bins=20, alpha=0.7, color="blue", edgecolor="black")
plt.axvline(
    x=np.mean(cers), color="red", linestyle="--", label=f"Mean CER: {np.mean(cers):.3f}"
)
plt.axvline(
    x=np.median(cers),
    color="green",
    linestyle="--",
    label=f"Median CER: {np.median(cers):.3f}",
)
plt.title("Distribution of Character Error Rates on Test Set")
plt.xlabel("Character Error Rate (CER)")
plt.ylabel("Number of Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Plot character-specific error rates
plt.figure(figsize=(14, 8))
char_error_df = pd.DataFrame(
    {"Character": list(char_errors.keys()), "Error Rate": list(char_errors.values())}
)
char_error_df = char_error_df.sort_values("Error Rate", ascending=False)

sns.barplot(x="Character", y="Error Rate", data=char_error_df.head(15))
plt.title("Top 15 Characters with Highest Error Rates")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 5. Confusion Matrix for Most Common Characters
plt.figure(figsize=(16, 14))
# Select top N most frequent characters for visualization
char_freq = Counter(all_true_chars)
top_chars = [char for char, _ in char_freq.most_common(15)]
top_indices = [char_to_idx[char] for char in top_chars]

# Create a subset of the confusion matrix for these characters
conf_subset = conf_matrix_norm[np.ix_(top_indices, top_indices)]

# Create a heatmap
sns.heatmap(
    conf_subset,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=top_chars,
    yticklabels=top_chars,
)
plt.title("Confusion Matrix for Top 15 Most Common Characters")
plt.xlabel("Predicted Character")
plt.ylabel("True Character")
plt.tight_layout()
plt.show()

# 6. Top Confused Character Pairs
plt.figure(figsize=(14, 8))
# Create a dataframe of the top confused pairs
top_confusions = confusion_counts[:15]  # Top 15 confused pairs
conf_df = pd.DataFrame(
    top_confusions, columns=["True", "Predicted", "Count", "Percentage"]
)
conf_df["Pair"] = conf_df["True"] + " → " + conf_df["Predicted"]

# Plot the top confusions
sns.barplot(x="Pair", y="Percentage", data=conf_df)
plt.title("Top 15 Most Confused Character Pairs")
plt.xticks(rotation=45)
plt.ylabel("Confusion Percentage (%)")
plt.tight_layout()
plt.show()

# 7. Error Analysis - Relationship Between Text Length and CER
plt.figure()
text_lengths = [len(text) for text in true_texts]
plt.scatter(text_lengths, cers, alpha=0.7)
plt.title("Relationship Between Text Length and Character Error Rate")
plt.xlabel("Text Length (Characters)")
plt.ylabel("Character Error Rate (CER)")
plt.grid(True)

# Add trend line
z = np.polyfit(text_lengths, cers, 1)
p = np.poly1d(z)
plt.plot(
    sorted(text_lengths),
    p(sorted(text_lengths)),
    "r--",
    label=f"Trend: y={z[0]:.6f}x+{z[1]:.6f}",
)
plt.legend()
plt.tight_layout()
plt.show()

# 8. Examples of Recognition Results
plt.figure(figsize=(15, 10))
# Choose a few random samples to display
sample_indices = np.random.choice(
    range(len(true_texts)), min(5, len(true_texts)), replace=False
)

for i, idx in enumerate(sample_indices):
    plt.subplot(5, 1, i + 1)
    plt.imshow(np.reshape(test_X[idx], (32, 732)), cmap="gray")
    plt.axis("off")
    plt.title(
        f"True: '{true_texts[idx]}' | Predicted: '{predicted_texts[idx]}' | CER: {cers[idx]:.3f}"
    )
plt.tight_layout()
plt.show()

# 9. Boxplot of  CERgrouped by text length buckets
plt.figure(figsize=(12, 8))
# Create length buckets
length_bins = [0, 5, 10, 15, 20, float("inf")]
length_labels = ["1-5", "6-10", "11-15", "16-20", "21+"]
length_categories = []

for length in text_lengths:
    for i, upper in enumerate(length_bins[1:]):
        if length <= upper:
            length_categories.append(length_labels[i])
            break

# Create a dataframe for visualization
cer_length_df = pd.DataFrame({"CER": cers, "Length Category": length_categories})

# Create boxplot
sns.boxplot(x="Length Category", y="CER", data=cer_length_df)
plt.title("Character Error Rate Distribution by Text Length")
plt.xlabel("Text Length Category")
plt.ylabel("Character Error Rate (CER)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Character frequency vs error rate
plt.figure(figsize=(14, 8))
# Calculate character frequencies
char_freq_counts = Counter(all_true_chars)
total_chars = len(all_true_chars)
char_freq_percentage = {
    char: count / total_chars * 100 for char, count in char_freq_counts.items()
}

# Create dataframe for visualization
char_analysis_df = pd.DataFrame(
    {
        "Character": list(char_errors.keys()),
        "Error Rate": list(char_errors.values()),
        "Frequency (%)": [
            char_freq_percentage.get(char, 0) for char in char_errors.keys()
        ],
    }
)

# Plot scatter with size proportional to frequency
plt.figure(figsize=(12, 8))
plt.scatter(
    char_analysis_df["Frequency (%)"],
    char_analysis_df["Error Rate"],
    s=100,
    alpha=0.7,
    edgecolors="w",
    linewidth=0.5,
)

# Add character labels to points
for i, row in char_analysis_df.iterrows():
    plt.annotate(
        row["Character"],
        (row["Frequency (%)"], row["Error Rate"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.xlabel("Character Frequency (%)")
plt.ylabel("Error Rate")
plt.title("Character Error Rate vs. Frequency")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# 11. Cyrillic-specific confusion matrix
cyrillic_pairs = [
    ("о", "а"),
    ("и", "н"),
    ("ш", "щ"),
    ("р", "п"),
    ("з", "э"),
    ("ь", "ъ"),
    ("б", "в"),
    ("ч", "ц"),
    ("д", "у"),
]

cyrillic_chars = sorted(set([char for pair in cyrillic_pairs for char in pair]))

available_chars = [char for char in cyrillic_chars if char in all_chars]

if len(available_chars) > 0:
    # Get indices for these characters
    cyrillic_indices = [
        char_to_idx[char] for char in available_chars if char in char_to_idx
    ]

    # Create a subset confusion matrix
    cyrillic_conf = conf_matrix_norm[np.ix_(cyrillic_indices, cyrillic_indices)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cyrillic_conf,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[all_chars[i] for i in cyrillic_indices],
        yticklabels=[all_chars[i] for i in cyrillic_indices],
    )
    plt.title("Confusion Matrix for Commonly Confused Cyrillic Characters")
    plt.xlabel("Predicted Character")
    plt.ylabel("True Character")
    plt.tight_layout()
    plt.show()

# 12. Learning Rate During Training
plt.figure()
plt.plot(history["learning_rate"], "b-")
plt.title("Learning Rate Schedule During Training")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n--- Character Error Rate (CER) Statistics ---")
print(f"Mean CER: {np.mean(cers):.3%}")
print(f"Median CER: {np.median(cers):.3%}")
print(f"Min CER: {np.min(cers):.3%}")
print(f"Max CER: {np.max(cers):.3%}")

print("\n--- Top Confused Character Pairs ---")
for i, (true_char, pred_char, count, percentage) in enumerate(confusion_counts[:8]):
    print(
        f'"{true_char}" and "{pred_char}" ({percentage:.1f}% of all errors for "{true_char}")'
    )
