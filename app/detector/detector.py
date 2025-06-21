import os
from pathlib import Path
from craft.craft import CRAFT
from craft.craft_utils import adjustResultCoordinates, getDetBoxes
from craft.file_utils import saveResult
from craft.finetune import copyStateDict
from craft.imgproc import loadImage, normalizeMeanVariance, resize_aspect_ratio
import torch
from torch.autograd import Variable
import cv2


class GroundTruthGenerator:
    def __init__(self):
        """Initialize the Ground Truth Generator.
        
        Args:
            output_dir (str): Directory to save ground truth files
        """
        print("Loading CRAFT model...")
        self.craft_model = CRAFT()
        craft_model_path = "model/craft_mlt_25k.pth"

        self.craft_model.load_state_dict(copyStateDict(torch.load(craft_model_path, map_location='cpu')))
        self.craft_model.eval()
    
    
    def generate_from_detection(self, image_path, output_dir, text_threshold=0.7, 
                                link_threshold=0.4, low_text=0.4, 
                                canvas_size=1280, mag_ratio=1.5):
        """Generate ground truth file by detecting text regions in the image.
        
        Args:
            image_path (str): Path to the source image
            model: CRAFT text detection model
            text_threshold (float): Text confidence threshold
            link_threshold (float): Link confidence threshold
            low_text (float): Text low-bound score
            canvas_size (int): Image size for inference
            mag_ratio (float): Image magnification ratio
            
        Returns:
            str: Path to the generated ground truth file
            list: Detected polygons
        """
        os.makedirs(output_dir, exist_ok=True)
        image = loadImage(image_path)
        
        boxes, polys = self._detect_text(
            self.craft_model, image, text_threshold, link_threshold, low_text,
            canvas_size, mag_ratio
        )

        saveResult(image_path, image[:,:,::-1], polys, dirname=output_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        filepath = Path(f"{output_dir}/{base_name}_gt.txt")
        return filepath, polys
    

    
    def _detect_text(self, net, image, text_threshold, link_threshold, 
                     low_text, canvas_size, mag_ratio):
        """Detect text regions in the image using CRAFT model."""
        img_resized, target_ratio, _ = resize_aspect_ratio(
            image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        
        with torch.no_grad():
            y, _ = net(x)
        
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)

        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: 
                polys[k] = boxes[k]
        return boxes, polys
