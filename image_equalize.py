import cv2

# Load your image
img = cv2.imread('your_image.jpg')

# 1. Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. Split the HSV channels
h, s, v = cv2.split(hsv)

# 3. Perform histogram equalization on the V (Value) channel
v_equalized = cv2.equalizeHist(v)

# 4. Merge the channels back together
hsv_equalized = cv2.merge((h, s, v_equalized))

# 5. Convert back to BGR (RGB format for OpenCV)
final_img = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

# 6. Save the final result
cv2.imwrite('output_equalized.jpg', final_img)

print("Histogram equalization complete and saved as 'output_equalized.jpg'.")
