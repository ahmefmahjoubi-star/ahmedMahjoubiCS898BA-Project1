import cv2

# Load the normalized image from Part 2
img = cv2.imread('normalized_color_image.jpg')

# Convert to grayscale (required for thresholding)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1. OTSU'S GLOBAL THRESHOLDING ---
ret, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Apply mask: keeps color where mask is white
otsu_foreground = cv2.bitwise_and(img, img, mask=otsu_mask)

cv2.imwrite('otsu_binary_mask.jpg', otsu_mask)
cv2.imwrite('otsu_foreground_extraction.jpg', otsu_foreground)

# --- 2. ADAPTIVE THRESHOLDING ---
adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Apply mask: keeps color where mask is white
adaptive_foreground = cv2.bitwise_and(img, img, mask=adaptive_mask)

cv2.imwrite('adaptive_binary_mask.jpg', adaptive_mask)
cv2.imwrite('adaptive_foreground_extraction.jpg', adaptive_foreground)

print("Part 3 Success! All masks and extractions saved.")