import cv2
import numpy as np

# 1. Load the manual ground truth mask (read as grayscale)
# Ensure you saved your painted image exactly as 'ground_truth.jpg'
gt_img = cv2.imread('ground_truth.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Load the three algorithm predicted masks
otsu_img = cv2.imread('otsu_binary_mask.jpg', cv2.IMREAD_GRAYSCALE)
adapt_img = cv2.imread('adaptive_binary_mask.jpg', cv2.IMREAD_GRAYSCALE)
kmeans_img = cv2.imread('kmeans_mask.jpg', cv2.IMREAD_GRAYSCALE)

# Function to calculate IoU and Dice metrics
def evaluate_segmentation(gt, pred, method_name):
    # Convert images to boolean arrays (True for white pixels, False for black)
    gt_bool = gt > 127
    pred_bool = pred > 127
    
    # Calculate Intersection (A ∩ B) and Union (A ∪ B)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0.0
    
    # Calculate Dice Coefficient: 2 * |A ∩ B| / (|A| + |B|)
    dice = (2.0 * intersection) / (gt_bool.sum() + pred_bool.sum()) if (gt_bool.sum() + pred_bool.sum()) != 0 else 0.0
    
    # Print the results
    print(f"--- {method_name} ---")
    print(f"IoU (Jaccard Index): {iou:.4f}")
    print(f"Dice Coefficient:    {dice:.4f}\n")

# 3. Execute the comparisons
print("Evaluating Segmentation Methods against Ground Truth...\n")
evaluate_segmentation(gt_img, otsu_img, "Otsu's Global Thresholding")
evaluate_segmentation(gt_img, adapt_img, "Adaptive Thresholding")
evaluate_segmentation(gt_img, kmeans_img, "K-Means Clustering")