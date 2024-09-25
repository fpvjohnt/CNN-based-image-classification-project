import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image

# --- Step 1: Define transformations and load custom dataset ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
    transforms.RandomHorizontalFlip(),  # Data augmentation: Random horizontal flip
    transforms.RandomRotation(10),  # Data augmentation: Slight rotation
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB values
])

# Update the path to the dataset based on your folder structure
data_dir = 'D:/ImageProcess/data'

# Only include directories that contain images (filter out directories like cifar-10-batches-py)
valid_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Ensure only valid folders are passed to ImageFolder
if not valid_dirs:
    print("No valid image directories found!")
else:
    # Load custom dataset from folder structure
    train_dataset = ImageFolder(root=data_dir, transform=transform)

    # Create DataLoader for the dataset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Get the class names (folder names)
    class_names = train_dataset.classes  # ['cactus', 'fern', 'rose', 'sunflower']
    print(f"Classes: {class_names}")

    # --- Step 2: Define the CNN architecture ---
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 128 channels * 4x4 image size after pooling
            self.fc2 = nn.Linear(256, 4)  # 4 output classes (sunflower, rose, fern, cactus)

        def forward(self, x):
            # Apply convolutions followed by ReLU and pooling
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the CNN model
    model = SimpleCNN()

    # --- Step 3: Define the loss function and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Step 4: Training the CNN model ---
    epochs = 10  # Set the number of epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)  # Move model to GPU if available

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get inputs and labels, move them to the device (GPU if available)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print loss (every 10 mini-batches)
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10}")
                running_loss = 0.0

    print('Finished Training')

    # --- Step 5: Test the model with the sunflower.jpeg image in the root directory ---
    def preprocess_custom_image(image_path):
        """Preprocesses a custom image to fit the model input size."""
        transform_custom = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize image to 32x32 pixels
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB values
        ])
        
        # Load and transform the image
        image = Image.open(image_path)
        image = transform_custom(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

    # Path to the sunflower.jpeg file in the root directory
    test_image_path = 'D:/ImageProcess/sunflower.jpeg'  # Ensure this is the correct path to sunflower.jpeg
    test_image = preprocess_custom_image(test_image_path)
    test_image = test_image.to(device)

    # Run the image through the model and get predictions
    model.eval()  # Set model to evaluation mode
    output = model(test_image)
    _, predicted_label = torch.max(output, 1)

    # Print the predicted class
    print(f'Predicted class for the sunflower.jpeg image: {class_names[predicted_label.item()]}')
