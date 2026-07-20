import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import your network blueprint!
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

# Isolate the Testing Set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

test_tensor = torch.tensor(X_test)
test_labels = y_test

print("\n2. Loading the Optimized Model...")
tuned_model = FishCNN(num_classes=6)
tuned_model.load_state_dict(torch.load('best_tuned_model.pth'))
tuned_model.eval()

print("\n3. Generating Predictions...")
with torch.no_grad():
    outputs = tuned_model(test_tensor)
    _, preds = torch.max(outputs, 1)

print("\n4. Plotting and Saving Confusion Matrix...")
# Calculate the matrix mathematically
cm = confusion_matrix(test_labels, preds.numpy())

# Set up the visual plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

# Add labels and title
plt.title('Optimized Model - Test Set Confusion Matrix', pad=20, fontsize=14)
plt.ylabel('Actual Fish Species', fontsize=12)
plt.xlabel('Predicted Fish Species', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the physical file
plt.savefig('confusion_matrix.png')
print(" -> Success! Confusion matrix saved to 'confusion_matrix.png'")