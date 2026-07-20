import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your network blueprint from your other file!
from network import FishCNN

print("1. Loading the Unseen Testing Data...")
DATA_DIR = "./Fish/Fish"
TARGET_SIZE = (128, 128)
images, labels = [], []
class_names = sorted(os.listdir(DATA_DIR))

for class_idx, folder_name in enumerate(class_names):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, TARGET_SIZE)
                img_transposed = np.transpose(img_resized, (2, 0, 1))
                img_normalized = img_transposed.astype(np.float32) / 255.0
                images.append(img_normalized)
                labels.append(class_idx)

X = np.array(images)
y = np.array(labels)

# Recreate the exact same split to isolate the Testing Set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Convert testing data to PyTorch Tensors
test_tensor = torch.tensor(X_test)
test_labels = y_test

print("\n2. Loading Saved Models...")
# Initialize blueprints
baseline_model = FishCNN(num_classes=6)
tuned_model = FishCNN(num_classes=6)

# Load the saved mathematical weights into the blueprints
baseline_model.load_state_dict(torch.load('baseline_model_weights.pth'))
tuned_model.load_state_dict(torch.load('best_tuned_model.pth'))

# Set both models to Evaluation Mode
baseline_model.eval()
tuned_model.eval()

print("\n3. Generating Predictions on the Test Set...")
with torch.no_grad(): # Turn off gradient tracking to save memory
    # Baseline Predictions
    baseline_outputs = baseline_model(test_tensor)
    _, baseline_preds = torch.max(baseline_outputs, 1)
    
    # Tuned Predictions
    tuned_outputs = tuned_model(test_tensor)
    _, tuned_preds = torch.max(tuned_outputs, 1)

print("\n=======================================================")
print("          BASELINE MODEL CLASSIFICATION REPORT         ")
print("=======================================================")
# The classification_report automatically calculates Precision, Recall, and F1
print(classification_report(test_labels, baseline_preds.numpy(), target_names=class_names))

print("\n=======================================================")
print("          OPTIMIZED MODEL CLASSIFICATION REPORT        ")
print("=======================================================")
print(classification_report(test_labels, tuned_preds.numpy(), target_names=class_names))