import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import pandas as pd

# ---------------------------
# 1. Define the Autoencoder
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 512, 512)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (B, 64, 256, 256)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 256, 256)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (B, 128, 128, 128)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, 128, 128)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (B, 256, 64, 64)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (B, 512, 64, 64)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # (B, 256, 64, 64)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B, 256, 128, 128)

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 128, 128)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B, 128, 256, 256)

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 256, 256)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B, 64, 512, 512)

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 512, 512)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (B, 32, 1024, 1024)

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # (B, 1, 1024, 1024)
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------
# 2. Define the Dataset
# ---------------------------
class NPYDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the .npy images.
        """
        self.root_dir = root_dir
        self.npy_files = [file for file in os.listdir(root_dir) if file.endswith('.npy')]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.root_dir, self.npy_files[idx])
        image = np.load(npy_path)  # Shape: (512, 512) or (H, W)

        # Normalize the image to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Add channel dimension: (1, 512, 512)
        image = np.expand_dims(image, axis=0)

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)

        return image

# ---------------------------
# 3. PSNR Calculation Function
# ---------------------------
def calculate_psnr(original, upscaled, max_pixel_value=1.0):
    mse = np.mean((original - upscaled) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr

# ---------------------------
# 4. Saving and Displaying Outputs
# ---------------------------
def save_outputs(model, dataset, device, npy_dir='output_npy_images', png_dir='output_png_images', num_display=10):
    model.eval()

    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    psnr_data = []  # List to store PSNR values for all outputs
    psnr_data_png = []  # List to store PSNR values for 10 .png outputs

    with torch.no_grad():
        psnr_values = []  # To calculate average PSNR
        for idx in range(len(dataset)):
            img = dataset[idx].unsqueeze(0).to(device)  # (1, 1, 512, 512)
            
            # Run the autoencoder to get the upscaled image (1024x1024)
            output = model(img)  # (1, 1, 1024, 1024)
            output = output.squeeze(0).squeeze(0).cpu().numpy()  # (1024, 1024)

            # Resize the original image to 1024x1024 to calculate PSNR
            original_resized = F.interpolate(img, size=(1024, 1024), mode='bilinear', align_corners=False)
            original_resized = original_resized.squeeze(0).squeeze(0).cpu().numpy()  # (1024, 1024)

            # Calculate PSNR
            psnr_value = calculate_psnr(original_resized, output)
            psnr_values.append(psnr_value)
            psnr_data.append({"Image Index": idx, "PSNR (dB)": psnr_value})

            # Save the upscaled image as a .npy file
            output_path_npy = os.path.join(npy_dir, f'upscaled_{idx}.npy')
            np.save(output_path_npy, output)

            # Save 10 images in .png format with ground truth
            if idx < num_display:
                output_path_png = os.path.join(png_dir, f'upscaled_{idx}.png')
                original_path_png = os.path.join(png_dir, f'original_{idx}.png')

                plt.imsave(output_path_png, output, cmap='gray')
                plt.imsave(original_path_png, original_resized, cmap='gray')

                psnr_data_png.append({"Image Index": idx, "PSNR (dB)": psnr_value})

        # Save PSNR values for all outputs in an Excel file
        psnr_df = pd.DataFrame(psnr_data)
        psnr_df.to_excel('psnr_values_all_outputs.xlsx', index=False)
        print(f'PSNR values saved to psnr_values_all_outputs.xlsx')

        # Save PSNR values for the 10 .png outputs in a separate Excel file
        psnr_df_png = pd.DataFrame(psnr_data_png)
        psnr_df_png.to_excel('psnr_values_png_outputs.xlsx', index=False)
        print(f'PSNR values for 10 PNG outputs saved to psnr_values_png_outputs.xlsx')

        # Calculate and print average PSNR for all outputs
        avg_psnr = np.mean(psnr_values)
        print(f'Average PSNR for all images: {avg_psnr:.2f} dB')

# ---------------------------
# 5. Main Execution
# ---------------------------
def main():
    # Parameters
    npy_dir = 'path_to_512_images'  # Replace with your .npy images directory
    output_npy_dir = 'output_npy_images'
    output_png_dir = 'output_png_images'
    batch_size = 8
    num_epochs = 2  # Increase this number for longer training
    learning_rate = 1e-6
    gradient_clip = 1.0

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize Dataset and DataLoader
    train_dataset = NPYDataset(root_dir=npy_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize Model, Loss Function, and Optimizer
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Autoencoder
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            img = data.to(device)  # (B, 1, 512, 512)

            # Forward pass
            output = model(img)  # (B, 1, 1024, 1024)

            # Resize target image to match output size
            img_resized = F.interpolate(img, size=(1024, 1024), mode='bilinear', align_corners=False)

            # Compute loss
            loss = criterion(output, img_resized)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Training Completed.')

    # Save and display outputs
    save_outputs(
        model=model,
        dataset=train_dataset,
        device=device,
        npy_dir=output_npy_dir,
        png_dir=output_png_dir,
        num_display=10
    )

if __name__ == '__main__':
    main()
