import cv2
import numpy as np
import os

# Set the folder you want to process
subset_folder = 'subset_1' 
output_folder = 'edge_results'
os.makedirs(output_folder, exist_ok=True)

# Prewitt Kernels
kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

for filename in os.listdir(subset_folder):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(subset_folder, filename), cv2.IMREAD_GRAYSCALE)
        
        # 1. Sobel
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)
        
        # 2. Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        
        # 3. Canny
        canny = cv2.Canny(img, 100, 200)
        
        # 4. Prewitt
        prewitt_x = cv2.filter2D(img, -1, kernel_x)
        prewitt_y = cv2.filter2D(img, -1, kernel_y)
        prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        
        # Save results
        cv2.imwrite(os.path.join(output_folder, f'sobel_{filename}'), sobel)
        cv2.imwrite(os.path.join(output_folder, f'laplacian_{filename}'), laplacian)
        cv2.imwrite(os.path.join(output_folder, f'canny_{filename}'), canny)
        cv2.imwrite(os.path.join(output_folder, f'prewitt_{filename}'), prewitt)

print("Edge detection complete. Files saved in 'edge_results' folder.")