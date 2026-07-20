import cv2
import numpy as np
import os

# Define paths - ensuring we do not overwrite any existing files
input_image_path = 'HW1_IMG_CS898BA (1).png'
output_dir = 'Preprocessed_Channels'

# Create an output directory if it doesn't exist to store the new images safely
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the target image
img_bgr = cv2.imread(input_image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not locate {input_image_path}. Please ensure it is in the same directory as this script.")

# ==========================================
# Channel A: Baseline
# The original, unmodified RGB image
# ==========================================
channel_a_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ==========================================
# Channel B: Perceptual Transformation
# Normalized according to the V channel in HSV
# ==========================================
# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Split the channels
h, s, v = cv2.split(img_hsv)

# Normalize ONLY the V (Value/Brightness) channel
v_normalized = cv2.normalize(v, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Merge back the channels and convert to RGB format for the network
img_hsv_normalized = cv2.merge([h, s, v_normalized])
channel_b_rgb = cv2.cvtColor(img_hsv_normalized, cv2.COLOR_HSV2RGB)

# ==========================================
# Channel C: Statistical Contrast Normalization
# RGB space, individual channels independently normalized
# ==========================================
# Split the original RGB image into individual channels
r, g, b = cv2.split(channel_a_rgb)

# Normalize Red, Green, and Blue independently
r_norm = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
g_norm = cv2.normalize(g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
b_norm = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Merge the independently normalized channels back together
channel_c_rgb = cv2.merge([r_norm, g_norm, b_norm])

# ==========================================
# Save the results to verify the transformations 
# (Converting back to BGR for OpenCV saving)
# ==========================================
cv2.imwrite(os.path.join(output_dir, 'Channel_A_Baseline.png'), cv2.cvtColor(channel_a_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'Channel_B_V_Normalized.png'), cv2.cvtColor(channel_b_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'Channel_C_RGB_Normalized.png'), cv2.cvtColor(channel_c_rgb, cv2.COLOR_RGB2BGR))

print("Pipeline complete. Channels A, B, and C have been generated and safely saved to the 'Preprocessed_Channels' folder.")
print("These RGB matrices are now ready to be converted to PyTorch tensors and fed into the Mask2Former inference engine.")