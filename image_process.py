import cv2

# Load the image
image_path = 'your_image.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load '{image_path}'. Check the filename.")
else:
    # 1. Convert to Greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output_grey.jpg', grey)

    # 2. Convert to Binary (Simple threshold)
    _, binary = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('output_binary.jpg', binary)

    # 3. Convert to Color Spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    cv2.imwrite('output_hsv.jpg', hsv)
    cv2.imwrite('output_lab.jpg', lab)
    cv2.imwrite('output_hls.jpg', hls)

    print("Successfully processed and saved all images.")
