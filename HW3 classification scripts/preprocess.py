import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Define Paths and Parameters
# Updated to correctly navigate into the nested Fish folder
DATA_DIR = "./Fish/Fish"  
TARGET_SIZE = (128, 128)  

images = []
labels = []
class_names = []

print("Loading dataset...")
if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset folder '{DATA_DIR}' not found. Please verify the folder name.")
else:
    # Read the subfolders inside Fish/Fish
    for folder_name in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            class_names.append(folder_name)
            class_idx = len(class_names) - 1
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize and normalize to [0, 1] range
                    img_resized = cv2.resize(img, TARGET_SIZE)
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    images.append(img_normalized)
                    labels.append(class_idx)

    X = np.array(images)
    y = np.array(labels)
    print(f"Loaded {len(X)} images across {len(class_names)} classes.")

    # 2. Stratified Split (70% Train, 15% Val, 15% Test)
    # The random_state ensures you get the exact same split every time you run it
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")

    # 3. Data Augmentation structural function
    def augment_image(image):
        augmented = [image]
        
        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            augmented.append(cv2.flip(image, 1))
            
        # Minor Rotation (Between -15 and 15 degrees)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w//2, h//2), np.random.uniform(-15, 15), 1.0)
        augmented.append(cv2.warpAffine(image, matrix, (w, h)))
        
        return augmented

    print("Data pipeline setup complete successfully!")