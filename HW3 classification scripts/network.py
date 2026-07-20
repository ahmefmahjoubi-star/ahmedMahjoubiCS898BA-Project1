import torch
import torch.nn as nn
import torch.nn.functional as F

print("Building the Convolutional Neural Network using PyTorch...")

class FishCNN(nn.Module):
    def __init__(self, num_classes):
        super(FishCNN, self).__init__()
        # 1. First Convolutional Block (32 filters) + Max-Pooling
        # Note: PyTorch reads image channels first (Color, Height, Width)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Second Convolutional Block (64 filters) + Max-Pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 3. Third Convolutional Block (128 filters) + Max-Pooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 4. Dense Layers
        # A 128x128 image halved 3 times by pooling becomes 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply Convolution -> ReLU Activation -> Max Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Pass through fully connected hidden layer
        x = F.relu(self.fc1(x))
        # Final softmax output layer
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Initialize the model using the 6 fish classes we found during preprocessing
model = FishCNN(num_classes=6)

# Display the network structure
print(model)
print("Network architecture built successfully!")