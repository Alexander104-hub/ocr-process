import os
import time
import argparse
from app.craft.data.gaussian import GaussianBuilder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import cv2
from collections import OrderedDict

from craft import CRAFT
import imgproc

from torch.utils.tensorboard import SummaryWriter

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


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
        enlarge_affinity=0.5
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
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                gt_path = os.path.join(root_dir, os.path.splitext(file)[0] + '_gt.txt')
                if os.path.exists(gt_path):
                    self.img_names.append(file)
                    self.gt_paths.append(gt_path)
        print(f"Found {len(self.img_names)} image-gt pairs")

    def __len__(self):
        return len(self.img_names)

    def load_gt_boxes(self, gt_path):
        boxes = []
        words = []
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    box = [float(val) for val in parts[:8]]
                    box = np.array(box, np.float32).reshape(4, 2)
                    boxes.append(box)

                    word = ','.join(parts[8:])
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
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        h, w, _ = image.shape

        word_bboxes, words = self.load_gt_boxes(gt_path)

        if self.mag_ratio != 1.0 and len(word_bboxes) > 0:
            word_bboxes = word_bboxes * ratio

        confidence_mask = np.ones((h, w), dtype=np.float32)

        if len(word_bboxes) > 0:
            word_level_char_bbox, horizontal_text_bools = self.prepare_word_char_boxes(word_bboxes, words)

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
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded_image = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
            padded_image[:new_h, :new_w, :] = resized_image
            
            resized_region = cv2.resize(region_score, (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR)
            resized_affinity = cv2.resize(affinity_score, (new_w, new_h),
                                      interpolation=cv2.INTER_LINEAR)
            resized_confidence = cv2.resize(confidence_mask, (new_w, new_h),
                                        interpolation=cv2.INTER_NEAREST)
            
            padded_region = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            padded_affinity = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            padded_confidence = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            
            padded_region[:new_h, :new_w] = resized_region
            padded_affinity[:new_h, :new_w] = resized_affinity
            padded_confidence[:new_h, :new_w] = resized_confidence
            
            image = padded_image
            region_score = padded_region
            affinity_score = padded_affinity
            confidence_mask = padded_confidence

        region_score = cv2.resize(region_score, 
                               (self.output_size // 2, self.output_size // 2), 
                               interpolation=cv2.INTER_CUBIC)
        affinity_score = cv2.resize(affinity_score, 
                                 (self.output_size // 2, self.output_size // 2), 
                                 interpolation=cv2.INTER_CUBIC)
        confidence_mask = cv2.resize(confidence_mask, 
                                  (self.output_size // 2, self.output_size // 2), 
                                  interpolation=cv2.INTER_NEAREST)

        normalized_image = imgproc.normalizeMeanVariance(image, mean=self.mean, variance=self.variance)
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
            'image': image_tensor,
            'target': target_tensor,
            'confidence_mask': confidence_mask_tensor,
            'image_path': img_path,
            'original_shape': (original_h, original_w),
            'bboxes': boxes_tensor,
            'original_image': image
        }

        if self.transform:
            sample = self.transform(sample)
            
        return sample



def craft_collate_fn(batch):
    """
    Custom collate function for the CRAFT dataset.
    
    Args:
        batch: A list of samples from the dataset
        
    Returns:
        Batched data with tensors of appropriate shapes
    """
    images = []
    targets = []
    image_paths = []
    
    for sample in batch:
        images.append(sample['image'])
        targets.append(sample['target'])
        image_paths.append(sample['image_path'])

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    bboxes = [sample['bboxes'] for sample in batch]
    
    return {
        'image': images,
        'target': targets,
        'bboxes': bboxes,
        'image_path': image_paths
    }

class Maploss_v2(nn.Module):
    def __init__(self):
        super(Maploss_v2, self).__init__()
        
    def batch_image_loss(self, pred_score, label_score, neg_rto, n_min_neg):
        # positive_loss
        positive_pixel = (label_score > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = pred_score * positive_pixel
        
        negative_pixel = (label_score <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)
        negative_loss_region = pred_score * negative_pixel
        
        if positive_pixel_number != 0:
            if negative_pixel_number < neg_rto * positive_pixel_number:
                negative_loss = (
                    torch.sum(
                        torch.topk(
                            negative_loss_region.view(-1), n_min_neg, sorted=False
                        )[0]
                    )
                    / n_min_neg
                )
            else:
                negative_loss = torch.sum(
                    torch.topk(
                        negative_loss_region.view(-1),
                        int(neg_rto * positive_pixel_number),
                        sorted=False,
                    )[0]
                ) / (positive_pixel_number * neg_rto)
            positive_loss = torch.sum(positive_loss_region) / positive_pixel_number
        else:
            negative_loss = (
                torch.sum(
                    torch.topk(negative_loss_region.view(-1), n_min_neg, sorted=False)[0]
                )
                / n_min_neg
            )
            positive_loss = 0.0
        
        total_loss = positive_loss + negative_loss
        return total_loss
    
    def forward(
        self,
        region_scores_label,
        affinity_scores_label,
        region_scores_pre,
        affinity_scores_pre,
        mask,
        neg_rto,
        n_min_neg,
    ):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        assert (
            region_scores_label.size() == region_scores_pre.size()
            and affinity_scores_label.size() == affinity_scores_pre.size()
        )

        loss1 = loss_fn(region_scores_pre, region_scores_label)
        loss2 = loss_fn(affinity_scores_pre, affinity_scores_label)

        loss_region = torch.mul(loss1, mask)
        loss_affinity = torch.mul(loss2, mask)
        
        char_loss = self.batch_image_loss(
            loss_region, region_scores_label, neg_rto, n_min_neg
        )
        affi_loss = self.batch_image_loss(
            loss_affinity, affinity_scores_label, neg_rto, n_min_neg
        )

        return char_loss + affi_loss

def train(net, train_loader, criterion, optimizer, device, epoch, writer, neg_rto=3.0, n_min_neg=10000):
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = sample['image'].to(device)
        targets = sample['target'].to(device)

        optimizer.zero_grad()

        outputs, _ = net(images)

        region_scores_label = targets[:, 0:1, :, :]
        affinity_scores_label = targets[:, 1:2, :, :]

        outputs = outputs.permute(0, 3, 1, 2)
        
        region_scores_pre = outputs[:, 0:1, :, :]
        affinity_scores_pre = outputs[:, 1:2, :, :]

        mask = torch.ones_like(region_scores_label)

        loss = criterion(
            region_scores_label, 
            affinity_scores_label, 
            region_scores_pre, 
            affinity_scores_pre, 
            mask, 
            neg_rto, 
            n_min_neg
        )

        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            
            # Log to tensorboard
            step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train', losses.val, step)
            
            # Log example predictions and targets
            if i == 0 and epoch % 5 == 0:
                # Get first image, prediction, and target
                image = images[0].permute(1, 2, 0).detach().cpu().numpy()
                image = imgproc.denormalizeMeanVariance(image)
                
                # Get region score
                pred_region = region_scores_pre[0, 0].detach().cpu().numpy()
                target_region = region_scores_label[0, 0].detach().cpu().numpy()
                
                # Convert to heatmap image
                pred_heatmap = imgproc.cvt2HeatmapImg(pred_region)
                target_heatmap = imgproc.cvt2HeatmapImg(target_region)
                
                # Log to tensorboard
                writer.add_image('Train/Image', image.transpose(2, 0, 1), epoch)
                writer.add_image('Train/PredRegion', pred_heatmap.transpose(2, 0, 1), epoch)
                writer.add_image('Train/TargetRegion', target_heatmap.transpose(2, 0, 1), epoch)
    
    return losses.avg


def validate(net, val_loader, criterion, device, epoch, writer, neg_rto=3.0, n_min_neg=10000):
    net.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            # Get the inputs and targets
            images = sample['image'].to(device)
            targets = sample['target'].to(device)
            
            # Forward
            outputs, _ = net(images)
            
            # Prepare inputs for the loss function
            # Extract region and affinity scores from targets
            region_scores_label = targets[:, 0:1, :, :]
            affinity_scores_label = targets[:, 1:2, :, :]
            
            # Permute outputs to match targets format
            outputs = outputs.permute(0, 3, 1, 2)
            
            region_scores_pre = outputs[:, 0:1, :, :]
            affinity_scores_pre = outputs[:, 1:2, :, :]
            
            # Create mask (all ones for now)
            mask = torch.ones_like(region_scores_label)
            
            # Calculate loss
            loss = criterion(
                region_scores_label, 
                affinity_scores_label, 
                region_scores_pre, 
                affinity_scores_pre, 
                mask, 
                neg_rto, 
                n_min_neg
            )
            
            # Update statistics
            losses.update(loss.item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:
                print(f'Validation: [{epoch}][{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})')
    
    # Log to tensorboard
    writer.add_scalar('Loss/validation', losses.avg, epoch)
    
    # Log example predictions and targets for validation set
    if epoch % 5 == 0 and len(val_loader) > 0:
        with torch.no_grad():
            # Get validation sample
            sample = next(iter(val_loader))
            images = sample['image'].to(device)
            targets = sample['target'].to(device)
            
            # Get prediction
            outputs, _ = net(images)
            
            # Permute outputs to match targets format
            outputs = outputs.permute(0, 3, 1, 2)
            
            # Get first image, prediction, and target
            image = images[0].permute(1, 2, 0).detach().cpu().numpy()
            image = imgproc.denormalizeMeanVariance(image)
            
            # Get region score
            pred_region = outputs[0, 0].detach().cpu().numpy()
            target_region = targets[0, 0].detach().cpu().numpy()
            
            # Convert to heatmap image
            pred_heatmap = imgproc.cvt2HeatmapImg(pred_region)
            target_heatmap = imgproc.cvt2HeatmapImg(target_region)
            
            # Log to tensorboard
            writer.add_image('Val/Image', image.transpose(2, 0, 1), epoch)
            writer.add_image('Val/PredRegion', pred_heatmap.transpose(2, 0, 1), epoch)
            writer.add_image('Val/TargetRegion', target_heatmap.transpose(2, 0, 1), epoch)
    
    return losses.avg

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


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')
    epoch = state.get('epoch', 0)
    if epoch % 5 == 0:
        torch.save(state['state_dict'], f'model_epoch_{epoch}.pth')


def main():
    parser = argparse.ArgumentParser(description='CRAFT Text Detection Training')
    parser.add_argument('--data_path', default='./train_data/', type=str, help='folder path to training data')
    parser.add_argument('--val_split', default=0.15, type=float, help='validation split ratio')
    parser.add_argument('--pretrained_model', default='model/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for training')
    parser.add_argument('--checkpoint_path', default='model/save', type=str, help='path to save checkpoints')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--neg_rto', default=3.0, type=float, help='negative ratio for loss calculation')
    parser.add_argument('--n_min_neg', default=15000, type=int, help='minimum number of negative pixels')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and load pre-trained weights
    net = CRAFT()
    
    if args.pretrained_model:
        print(f'Loading weights from checkpoint ({args.pretrained_model})')
        net.load_state_dict(copyStateDict(torch.load(args.pretrained_model, map_location=device)))
    
    # Move model to device
    net = net.to(device)
    
    if torch.cuda.device_count() > 1 and args.cuda:
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    
    # Setup TensorBoard
    writer = SummaryWriter()
    
    # Define loss function and optimizer
    criterion = Maploss_v2()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    dataset = CRAFTDataset(args.data_path)
    
    # Handle small dataset case
    if len(dataset) < 2:
        print("Dataset too small for validation split. Using all data for training.")
        train_dataset = dataset
        val_dataset = dataset
    else:
        dataset_size = len(dataset)
        val_size = max(1, int(args.val_split * dataset_size))
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, collate_fn=craft_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, collate_fn=craft_collate_fn)

    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch {epoch}/{args.num_epochs - 1}')
        print('-' * 10)

        train_loss = train(net, train_loader, criterion, optimizer, device, epoch, writer, 
                          args.neg_rto, args.n_min_neg)

        val_loss = validate(net, val_loader, criterion, device, epoch, writer,
                           args.neg_rto, args.n_min_neg)

        scheduler.step(val_loss)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.checkpoint_path, f'checkpoint_epoch_{epoch}.pth'))

        torch.save(net.state_dict(), os.path.join(args.checkpoint_path, 'craft_finetune_last.pth'))
        
        if is_best:
            torch.save(net.state_dict(), os.path.join(args.checkpoint_path, 'craft_finetune_best.pth'))
    
    writer.close()
    print('Training completed!')

if __name__ == '__main__':
    import shutil
    main()
