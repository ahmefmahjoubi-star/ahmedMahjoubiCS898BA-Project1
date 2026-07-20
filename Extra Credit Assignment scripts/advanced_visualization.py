import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# ==========================================
# 1. Initialization & Metrics Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Mask2Former model weights...")
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model.to(device)

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()
    if total_pixels == 0:
        return 0.0
    return (2. * intersection) / total_pixels

def generate_overlay(original_img, mask, alpha=0.5):
    color_mask = np.zeros_like(original_img)
    color_mask[mask == 1] = [0, 255, 0] # Bright green overlay
    return cv2.addWeighted(original_img, 1 - alpha, color_mask, alpha, 0)

# ==========================================
# 2. Ground Truth Loading 
# ==========================================
ground_truth_path = 'ground_truth.jpg' 

if not os.path.exists(ground_truth_path):
    print(f"Error: Could not find '{ground_truth_path}'.")
    exit()

gt_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
_, gt_binary = cv2.threshold(gt_img, 127, 1, cv2.THRESH_BINARY)
gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB) # For plotting

# ==========================================
# 3. Inference and Evaluation Pipeline
# ==========================================
def evaluate_channel(image_path, channel_name):
    print(f"Evaluating {channel_name}...")
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    target_sizes = [img_rgb.shape[:2]]
    results = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)[0]
    panoptic_map = results['segmentation'].cpu().numpy()
    
    best_iou = 0
    best_mask = np.zeros_like(gt_binary)
    
    for segment_id in np.unique(panoptic_map):
        if segment_id == 0: 
            continue
        current_mask = (panoptic_map == segment_id).astype(np.uint8)
        
        if current_mask.shape != gt_binary.shape:
            current_mask = cv2.resize(current_mask, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        iou = calculate_iou(current_mask, gt_binary)
        if iou > best_iou:
            best_iou = iou
            best_mask = current_mask
            
    final_iou = best_iou
    final_dice = calculate_dice(best_mask, gt_binary)
    
    if img_rgb.shape[:2] != best_mask.shape:
        img_rgb = cv2.resize(img_rgb, (best_mask.shape[1], best_mask.shape[0]))
        
    overlay = generate_overlay(img_rgb, best_mask)
    return img_rgb, overlay, final_iou, final_dice

# Run evaluation and grab both original RGBs and overlays
img_a, overlay_a, iou_a, dice_a = evaluate_channel('Preprocessed_Channels/Channel_A_Baseline.png', 'Channel A')
img_b, overlay_b, iou_b, dice_b = evaluate_channel('Preprocessed_Channels/Channel_B_V_Normalized.png', 'Channel B')
img_c, overlay_c, iou_c, dice_c = evaluate_channel('Preprocessed_Channels/Channel_C_RGB_Normalized.png', 'Channel C')

# ==========================================
# 4. Generate the 7-Image Visualization Grid
# ==========================================
print("Generating final 7-image visualization grid...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: The Input Channels & Ground Truth
axes[0, 0].imshow(img_a)
axes[0, 0].set_title('1. Original Image (Baseline)')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_b)
axes[0, 1].set_title('2. V-Channel Normalized')
axes[0, 1].axis('off')

axes[0, 2].imshow(img_c)
axes[0, 2].set_title('3. RGB-Channel Normalized')
axes[0, 2].axis('off')

axes[0, 3].imshow(gt_rgb)
axes[0, 3].set_title('4. Ground Truth Mask')
axes[0, 3].axis('off')

# Row 2: The Mask2Former Segmentations
axes[1, 0].imshow(overlay_a)
axes[1, 0].set_title(f'Segmented Baseline\nIoU: {iou_a:.4f} | Dice: {dice_a:.4f}')
axes[1, 0].axis('off')

axes[1, 1].imshow(overlay_b)
axes[1, 1].set_title(f'Segmented V-Norm\nIoU: {iou_b:.4f} | Dice: {dice_b:.4f}')
axes[1, 1].axis('off')

axes[1, 2].imshow(overlay_c)
axes[1, 2].set_title(f'Segmented RGB-Norm\nIoU: {iou_c:.4f} | Dice: {dice_c:.4f}')
axes[1, 2].axis('off')

# Leave the last subplot empty to balance the 2x4 grid
axes[1, 3].axis('off')

plt.tight_layout()
final_image_name = 'Comprehensive_Evaluation_Plot.png'
plt.savefig(final_image_name, bbox_inches='tight', facecolor='white')
print(f"Visualization complete. Saved to '{final_image_name}'")