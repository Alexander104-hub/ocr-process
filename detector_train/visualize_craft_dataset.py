import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import random

import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset

from data import imgproc
from data.gaussian import GaussianBuilder


class CRAFTDataset(Dataset):
    def __init__(
        self,
        root_dir,
        output_size=768,
        mean=(0.485, 0.456, 0.406),
        variance=(0.229, 0.224, 0.225),
        transform=None,
        mag_ratio=1.5,
        gauss_init_size=800,
        gauss_sigma=40,
        enlarge_region=1.0,
        enlarge_affinity=0.5,
    ):
        self.root_dir = root_dir
        self.output_size = output_size
        self.mean = mean
        self.variance = variance
        self.transform = transform
        self.mag_ratio = mag_ratio

        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )

        self.img_names = []
        self.gt_paths = []
        for file in os.listdir(root_dir):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                gt_path = os.path.join(root_dir, os.path.splitext(file)[0] + "_gt.txt")
                if os.path.exists(gt_path):
                    self.img_names.append(file)
                    self.gt_paths.append(gt_path)
        print(f"Found {len(self.img_names)} image-gt pairs")

    def __len__(self):
        return len(self.img_names)

    def load_gt_boxes(self, gt_path):
        boxes = []
        words = []
        with open(gt_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    box = [float(val) for val in parts[:8]]
                    box = np.array(box, np.float32).reshape(4, 2)
                    boxes.append(box)

                    word = ",".join(parts[8:])
                    words.append(word)

        return np.array(boxes), words

    def prepare_word_char_boxes(self, word_bboxes, words):
        word_level_char_bbox = []
        horizontal_text_bools = []

        for i, box in enumerate(word_bboxes):
            word = words[i]
            num_chars = max(1, len(word))

            width = np.linalg.norm(box[1] - box[0])
            height = np.linalg.norm(box[3] - box[0])
            is_horizontal = width > height
            horizontal_text_bools.append(is_horizontal)

            if is_horizontal:
                char_boxes = []
                left_vec = (box[3] - box[0]) / num_chars
                right_vec = (box[2] - box[1]) / num_chars

                for j in range(num_chars):
                    tl = box[0] + left_vec * j
                    tr = box[0] + left_vec * (j + 1)
                    br = box[1] + right_vec * (j + 1)
                    bl = box[1] + right_vec * j
                    char_box = np.array([tl, tr, br, bl])
                    char_boxes.append(char_box)
            else:
                char_boxes = []
                top_vec = (box[1] - box[0]) / num_chars
                bottom_vec = (box[2] - box[3]) / num_chars

                for j in range(num_chars):
                    tl = box[0] + top_vec * j
                    tr = box[0] + top_vec * (j + 1)
                    br = box[3] + bottom_vec * (j + 1)
                    bl = box[3] + bottom_vec * j
                    char_box = np.array([tl, tr, br, bl])
                    char_boxes.append(char_box)

            word_level_char_bbox.append(np.array(char_boxes))

        return word_level_char_bbox, horizontal_text_bools

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.root_dir, img_name)
        gt_path = self.gt_paths[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w, _ = image.shape

        if self.mag_ratio != 1.0:
            target_size = max(original_h, original_w) * self.mag_ratio
            ratio = target_size / max(original_h, original_w)
            target_h, target_w = int(original_h * ratio), int(original_w * ratio)
            image = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        h, w, _ = image.shape

        word_bboxes, words = self.load_gt_boxes(gt_path)

        if self.mag_ratio != 1.0 and len(word_bboxes) > 0:
            word_bboxes = word_bboxes * ratio

        confidence_mask = np.ones((h, w), dtype=np.float32)

        if len(word_bboxes) > 0:
            word_level_char_bbox, horizontal_text_bools = self.prepare_word_char_boxes(
                word_bboxes, words
            )

            region_score = self.gaussian_builder.generate_region(
                h, w, word_level_char_bbox, horizontal_text_bools
            )
            affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
                h, w, word_level_char_bbox, horizontal_text_bools
            )
        else:
            region_score = np.zeros((h, w), dtype=np.float32)
            affinity_score = np.zeros((h, w), dtype=np.float32)
            word_level_char_bbox = []

        if h != self.output_size or w != self.output_size:
            scale = min(self.output_size / h, self.output_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized_image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            padded_image = np.zeros(
                (self.output_size, self.output_size, 3), dtype=np.uint8
            )
            padded_image[:new_h, :new_w, :] = resized_image

            resized_region = cv2.resize(
                region_score, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            resized_affinity = cv2.resize(
                affinity_score, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            resized_confidence = cv2.resize(
                confidence_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

            padded_region = np.zeros(
                (self.output_size, self.output_size), dtype=np.float32
            )
            padded_affinity = np.zeros(
                (self.output_size, self.output_size), dtype=np.float32
            )
            padded_confidence = np.zeros(
                (self.output_size, self.output_size), dtype=np.float32
            )

            padded_region[:new_h, :new_w] = resized_region
            padded_affinity[:new_h, :new_w] = resized_affinity
            padded_confidence[:new_h, :new_w] = resized_confidence

            image = padded_image
            region_score = padded_region
            affinity_score = padded_affinity
            confidence_mask = padded_confidence

        region_score = cv2.resize(
            region_score,
            (self.output_size // 2, self.output_size // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        affinity_score = cv2.resize(
            affinity_score,
            (self.output_size // 2, self.output_size // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        confidence_mask = cv2.resize(
            confidence_mask,
            (self.output_size // 2, self.output_size // 2),
            interpolation=cv2.INTER_NEAREST,
        )

        normalized_image = imgproc.normalizeMeanVariance(
            image, mean=self.mean, variance=self.variance
        )
        normalized_image = normalized_image.transpose(2, 0, 1)  # HWC to CHW

        image_tensor = torch.from_numpy(normalized_image).float()
        region_score_tensor = torch.from_numpy(region_score).float().unsqueeze(0)
        affinity_score_tensor = torch.from_numpy(affinity_score).float().unsqueeze(0)
        confidence_mask_tensor = torch.from_numpy(confidence_mask).float().unsqueeze(0)

        target_tensor = torch.cat([region_score_tensor, affinity_score_tensor], dim=0)

        if len(word_bboxes):
            boxes_tensor = torch.tensor(word_bboxes[0], dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 8), dtype=torch.float32)

        sample = {
            "image": image_tensor,
            "target": target_tensor,
            "confidence_mask": confidence_mask_tensor,
            "image_path": img_path,
            "original_shape": (original_h, original_w),
            "bboxes": boxes_tensor,
            "original_image": image,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class imgproc:
    @staticmethod
    def normalizeMeanVariance(
        in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
    ):
        # should be RGB order
        img = in_img.copy()
        img = img.astype(np.float32)
        img = img / 255.0
        img = (img - np.array(mean)) / np.sqrt(np.array(variance))
        return img

    @staticmethod
    def denormalizeMeanVariance(
        in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
    ):
        # should be RGB order
        img = in_img.copy()
        img = img * np.sqrt(np.array(variance)) + np.array(mean)
        img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def cvt2HeatmapImg(img):
        """
        Convert heatmap to RGB image
        """
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize_craft_sample(dataset, sample_idx=None):
    """
    Visualize a sample from the CRAFT dataset
    Args:
        dataset: CRAFT dataset instance
        sample_idx: Index of sample to visualize. If None, a random sample is chosen.
    """
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)

    sample = dataset[sample_idx]

    original_image = sample["original_image"]
    region_score = sample["target"][0].numpy()
    affinity_score = sample["target"][1].numpy()
    boxes = sample["bboxes"].numpy()
    image_path = sample["image_path"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(f"Original Image: {os.path.basename(image_path)}")
    axes[0, 0].axis("off")

    image_with_boxes = original_image.copy()

    axes[0, 1].imshow(image_with_boxes)
    axes[0, 1].set_title(f"Image with Bounding Boxes: {len(boxes)} boxes")
    axes[0, 1].axis("off")

    region_heatmap = imgproc.cvt2HeatmapImg(region_score)
    axes[1, 0].imshow(region_heatmap)
    axes[1, 0].set_title("Region Score Map")
    axes[1, 0].axis("off")

    affinity_heatmap = imgproc.cvt2HeatmapImg(affinity_score)
    axes[1, 1].imshow(affinity_heatmap)
    axes[1, 1].set_title("Affinity Score Map")
    axes[1, 1].axis("off")

    plt.suptitle(
        f"CRAFT Dataset Visualization - Sample {sample_idx}/{len(dataset)}", fontsize=16
    )
    plt.tight_layout()
    plt.show()


def main():
    root_dir = "./train_data2/"
    dataset = CRAFTDataset(root_dir=root_dir)
    print(f"Visualizing 5 random samples from dataset with {len(dataset)} samples")
    for _ in range(1):
        visualize_craft_sample(dataset)


if __name__ == "__main__":
    main()
