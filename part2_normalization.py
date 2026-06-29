import cv2

# 1. Load original image
img = cv2.imread('your_image.jpg')

# 2. Split channels
b, g, r = cv2.split(img)

# 3. Equalize
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# 4. Merge
normalized_img = cv2.merge((b_eq, g_eq, r_eq))

# 5. Save output
cv2.imwrite('normalized_color_image.jpg', normalized_img)
print("Part 2 Success! 'normalized_color_image.jpg' created.")