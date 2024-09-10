import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Define the Autoencoder model (same as before)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # Output: 256x256

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # Output: 128x128

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # Output: 64x64

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Output: 128x128

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Output: 256x256

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Output: 512x512

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # Output: 1024x1024

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # For grayscale images
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define a custom dataset for .npy images (same as before)
class NPYDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.npy_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.root_dir, self.npy_files[idx])
        image = np.load(npy_path)  # Load image in numpy array format
        image = np.expand_dims(image, axis=0)  # Add channel dimension (1, 512, 512)
        image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
        return image

# Directory where .npy images are stored (same as before)
npy_dir = 'path_to_512_images'

# Dataset and DataLoader (same as before)
train_dataset = NPYDataset(root_dir=npy_dir)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Instantiate the model, define the loss function and optimizer (same as before)
model = Autoencoder().cuda()  # Move model to GPU if available
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the autoencoder with resizing input to 1024x1024 for loss calculation
num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        img = data.cuda()  # Move input data to GPU if available
        
        # Forward pass
        output = model(img)
        
        # Resize input image to 1024x1024 for comparison
        img_resized = F.interpolate(img, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        loss = criterion(output, img_resized)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, save the output images and display 5 of them (same as before)
model.eval()
output_dir = 'output_npy_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with torch.no_grad():
    # Load images one by one from the dataset
    for idx, img in enumerate(train_dataset):
        img = img.unsqueeze(0).cuda()  # Add batch dimension
        output = model(img)
        output = output.squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
        
        # Save the output image as a .npy file
        output_path = os.path.join(output_dir, f'output_{idx}.npy')
        np.save(output_path, output)
        
        # Show the first 5 output images
        if idx < 5:
            plt.imshow(output[0], cmap='gray')  # Show only the first channel for grayscale
            plt.title(f'Upscaled Image {idx}')
            plt.show()
