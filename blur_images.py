import cv2
import glob

# This grabs all images currently in your folder
all_images = glob.glob('*.jpg')
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

for img_name in all_images:
    img = cv2.imread(img_name)
    for s in sigmas:
        # Kernel size must be odd and positive
        k = int(s * 4) * 2 + 1 
        blurred = cv2.GaussianBlur(img, (k, k), s)
        cv2.imwrite(f'blur_{s}_{img_name}', blurred)

print("Step 8 complete: Blurs saved.")