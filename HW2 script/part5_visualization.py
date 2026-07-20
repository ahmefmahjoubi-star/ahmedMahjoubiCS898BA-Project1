import cv2
import matplotlib.pyplot as plt

# 1. Load the color images (convert from BGR to RGB for matplotlib)
orig_img = cv2.imread('your_image.jpg') 
orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

norm_img = cv2.imread('normalized_color_image.jpg')
norm_rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

# 2. Load the 4 masks (as grayscale)
otsu_mask = cv2.imread('otsu_binary_mask.jpg', cv2.IMREAD_GRAYSCALE)
adapt_mask = cv2.imread('adaptive_binary_mask.jpg', cv2.IMREAD_GRAYSCALE)
kmeans_mask = cv2.imread('kmeans_mask.jpg', cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.imread('ground_truth.jpg', cv2.IMREAD_GRAYSCALE)

# 3. Set up the plot (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top Row
axes[0, 0].imshow(orig_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(norm_rgb)
axes[0, 1].set_title('Normalized Image')
axes[0, 1].axis('off')

axes[0, 2].imshow(gt_mask, cmap='gray')
axes[0, 2].set_title('Ground Truth Mask')
axes[0, 2].axis('off')

# Bottom Row
axes[1, 0].imshow(otsu_mask, cmap='gray')
axes[1, 0].set_title('Otsu Mask')
axes[1, 0].axis('off')

axes[1, 1].imshow(adapt_mask, cmap='gray')
axes[1, 1].set_title('Adaptive Mask')
axes[1, 1].axis('off')

axes[1, 2].imshow(kmeans_mask, cmap='gray')
axes[1, 2].set_title('K-Means Mask')
axes[1, 2].axis('off')

# 4. Save the layout
plt.tight_layout()
plt.savefig('comparison_plot.png')
print("Part 6 Success! Comparison plot saved.")