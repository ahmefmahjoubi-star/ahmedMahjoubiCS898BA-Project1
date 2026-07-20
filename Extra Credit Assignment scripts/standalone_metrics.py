import torch
import cv2
import numpy as np
import os
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# ==========================================
# 1. Mathematical Metrics Definitions
# ==========================================
def calculate_iou(pred_mask, gt_mask):
    """Calculates Intersection over Union (IoU) / Jaccard Index."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """Calculates Dice Coefficient (F1-Score)."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()
    if total_pixels == 0:
        return 0.0
    return (2. * intersection) / total_pixels

# ==========================================
# 2. Setup Model & Ground Truth
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading Mask2Former model weights...")

processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model.to(device)

ground_truth_path = 'ground_truth.jpg'

if not os.path.exists(ground_truth_path):
    print(f"Error: Ground truth file '{ground_truth_path}' not found in project directory.")
    exit()

# Load ground truth image and binarize
gt_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
_, gt_binary = cv2.threshold(gt_img, 127, 1, cv2.THRESH_BINARY)

# ==========================================
# 3. Automatic Channel Evaluation Pipeline
# ==========================================
channels = {
    "Channel A (Baseline)": "Preprocessed_Channels/Channel_A_Baseline.png",
    "Channel B (V-Normalized)": "Preprocessed_Channels/Channel_B_V_Normalized.png",
    "Channel C (RGB-Normalized)": "Preprocessed_Channels/Channel_C_RGB_Normalized.png"
}

print("\n" + "="*55)
print("   MASK2FORMER SEGMENTATION METRICS EVALUATION")
print("="*55)

for name, path in channels.items():
    if not os.path.exists(path):
        print(f"\nWarning: File '{path}' not found. Skipping...")
        continue
        
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    target_sizes = [img_rgb.shape[:2]]
    results = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)[0]
    panoptic_map = results['segmentation'].cpu().numpy()
    
    # Isolate best matching target object mask against ground truth
    best_iou = 0.0
    best_dice = 0.0
    
    for segment_id in np.unique(panoptic_map):
        if segment_id == 0:  # Background segment
            continue
        current_mask = (panoptic_map == segment_id).astype(np.uint8)
        
        if current_mask.shape != gt_binary.shape:
            current_mask = cv2.resize(current_mask, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        iou = calculate_iou(current_mask, gt_binary)
        if iou > best_iou:
            best_iou = iou
            best_dice = calculate_dice(current_mask, gt_binary)
            
    print(f"\nResults for {name}:")
    print(f"  * IoU / Jaccard Index : {best_iou:.4f}")
    print(f"  * Dice Coefficient   : {best_dice:.4f}")

print("\n" + "="*55 + "\n")