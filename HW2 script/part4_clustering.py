import cv2
import numpy as np

# 1. Load normalized image
img = cv2.imread('normalized_color_image.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. Reshape for K-Means (it needs a 2D array of pixels)
pixel_values = img_hsv.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 3. Define criteria and apply K-Means
K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret, label, center = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 4. Segment the image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_hsv.shape))

# 5. Isolate one cluster (The "Unknown Figure")
mask = (label.reshape(img.shape[:2]) == 0).astype(np.uint8) * 255
segmented_foreground = cv2.bitwise_and(img, img, mask=mask)

# 6. Save results
cv2.imwrite('kmeans_mask.jpg', mask)
cv2.imwrite('kmeans_segmentation.jpg', segmented_foreground)

print("Part 4 Success! K-Means clustering complete.")