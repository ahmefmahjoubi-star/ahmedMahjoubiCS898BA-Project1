import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Configuration
subset_folder = 'subset_1' # Ensure this folder exists with your images
output_plots = 'plots'
os.makedirs(output_plots, exist_ok=True)

# Prewitt Kernels
kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Get list of images
images = [f for f in os.listdir(subset_folder) if f.endswith('.jpg')]
random.shuffle(images)

# Process 6 random images for the plots
for i in range(6):
    img_name = images[i]
    img_path = os.path.join(subset_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform Techniques
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    canny = cv2.Canny(img, 100, 200)
    prewitt_x = cv2.filter2D(img, -1, kernel_x)
    prewitt_y = cv2.filter2D(img, -1, kernel_y)
    prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Input', 'Sobel', 'Laplacian', 'Canny', 'Prewitt']
    data = [img, sobel, laplacian, canny, prewitt]
    
    for j in range(5):
        axes[j].imshow(data[j], cmap='gray')
        axes[j].set_title(titles[j])
        axes[j].axis('off')
        
    plt.savefig(os.path.join(output_plots, f'plot_{i}.png'))
    plt.close()

print("Plotting complete. 6 plots saved in the 'plots' folder.")