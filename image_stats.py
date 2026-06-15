import cv2
import numpy as np
from scipy.stats import mode, skew

# Load the image (replace 'your_image.jpg' with your actual image file name)
# Note: OpenCV loads images in BGR (Blue, Green, Red) format by default.
image_path = 'your_image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image at '{image_path}'. Please check the file path.")
else:
    # Split the image into its individual B, G, and R channels
    channels = cv2.split(img)
    channel_names = ['Blue', 'Green', 'Red']
    
    print(f"--- Basic Image Statistics for '{image_path}' ---\n")
    
    # Loop through each channel and calculate/print the statistics
    for i, channel in enumerate(channels):
        print(f"[{channel_names[i]} Channel]")
        
        # Calculate statistics
        c_min = np.min(channel)
        c_max = np.max(channel)
        c_mean = np.mean(channel)
        c_median = np.median(channel)
        
        # scipy.stats.mode returns an object; we extract just the mode value
        c_mode = mode(channel, axis=None, keepdims=False).mode
        
        # scipy.stats.skew calculates the skewness
        c_skew = skew(channel, axis=None)
        
        c_range = np.ptp(channel) # ptp stands for "peak to peak" (max - min)
        c_std = np.std(channel)
        c_var = np.var(channel)
      
        # Print the formatted results
        print(f"  Min:                {c_min}")
        print(f"  Max:                {c_max}")
        print(f"  Average (Mean):     {c_mean:.4f}")
        print(f"  Median:             {c_median}")
        print(f"  Mode:               {c_mode}")
        print(f"  Skew:               {c_skew:.4f}")
        print(f"  Range:              {c_range}")
        print(f"  Standard Deviation: {c_std:.4f}")
        print(f"  Variance:           {c_var:.4f}\n")
