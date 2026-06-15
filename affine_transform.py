import cv2
import numpy as np

# List of your generated images
images = ['output_binary.jpg', 'output_equalized.jpg', 'output_grey.jpg', 
          'output_hls.jpg', 'output_hsv.jpg', 'output_lab.jpg', 'your_image.jpg']

# This loop processes each image
for i, img_name in enumerate(images):
    img = cv2.imread(img_name)
    if img is None: 
        print(f"Skipping {img_name}, file not found.")
        continue
        
    rows, cols = img.shape[:2]

    # --- Transformation 1: Rotation ---
    angle = 15 + (i * 10) 
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    res_rot = cv2.warpAffine(img, M_rot, (cols, rows))
    cv2.imwrite(f'transform_{i}_rot.jpg', res_rot)

    # --- Transformation 2: Scaling ---
    scale = 0.7 + (i * 0.03) # Unique scale per image
    M_scale = np.float32([[scale, 0, 0], [0, scale, 0]])
    res_scale = cv2.warpAffine(img, M_scale, (cols, rows))
    cv2.imwrite(f'transform_{i}_scale.jpg', res_scale)

print("Finished: 14 transformations created.")