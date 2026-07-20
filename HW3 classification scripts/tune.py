import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Import your untouched network blueprint
from network import FishCNN

print("1. Loading Data for Hyperparameter Tuning...")
DATA_DIR = "./Fish/Fish"
TARGET_SIZE = (128, 128)
images, labels = [], []

for class_idx, folder_name in enumerate(sorted(os.listdir(DATA_DIR))):
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Convert arrays to PyTorch Tensors (DataLoaders are built inside the loop for batch sizes)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))

# 2. Define the Tuning Grid (From Assignment Prompt)
learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [32, 64]
weight_decays = [1e-3, 1e-4] # L2 Regularization to combat overfitting

EPOCHS = 5 # Using 5 epochs per test to save time during grid search
best_val_loss = float('inf')
best_config = {}

print(f"\n2. Beginning Grid Search (Testing {len(learning_rates) * len(batch_sizes) * len(weight_decays)} combinations)...")

# 3. The Grid Search Loop
for b_size in batch_sizes:
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)
    
    for lr in learning_rates:
        for wd in weight_decays:
            print(f"\n-> Testing Config: Batch={b_size}, LR={lr}, L2_Decay={wd}")
            
            # Initialize a fresh brain and optimizer for every test
            model = FishCNN(num_classes=6)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            current_val_loss = 0.0
            
            # Quick training loop
            for epoch in range(EPOCHS):
                model.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # Validation phase
                model.eval()
                running_val_loss, total = 0.0, 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        running_val_loss += loss.item() * inputs.size(0)
                        total += targets.size(0)
                
                current_val_loss = running_val_loss / total
            
            print(f"   Final Validation Loss for config: {current_val_loss:.4f}")
            
            # 4. Save the Best Configuration
            if current_val_loss < best_val_loss:
                print("   *** New Best Model Found! Saving... ***")
                best_val_loss = current_val_loss
                best_config = {'Batch': b_size, 'LR': lr, 'L2_Decay': wd}
                torch.save(model.state_dict(), 'best_tuned_model.pth')

print("\n=========================================")
print("3. Grid Search Complete!")
print(f"Best Configuration: {best_config}")
print(f"Lowest Validation Loss: {best_val_loss:.4f}")
print("The best weights have been saved physically as 'best_tuned_model.pth'")
print("=========================================")