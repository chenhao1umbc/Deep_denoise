import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import os
import math

# Custom dataset class to handle both .tif images and .avi videos
class DenoiseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = []
        
        # Collect all valid files
        for file in os.listdir(data_dir):
            if file.endswith(('.tif', '.avi')):
                self.file_list.append(os.path.join(data_dir, file))
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        if file_path.endswith('.tif'):
            # Handle .tif files
            image = Image.open(file_path)
            image = np.array(image)
        else:
            # Handle .avi files - take first frame for now
            cap = cv2.VideoCapture(file_path)
            ret, image = cap.read()
            cap.release()
            if ret:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        # Add noise to create noisy-clean pairs
        noisy_image = add_noise(image)
        return noisy_image, image

# Simple UNet-style model for diffusion
class DiffusionModel(nn.Module):
    def __init__(self, n_steps=1000):
        super(DiffusionModel, self).__init__()
        self.n_steps = n_steps
        
        # Define beta schedule
        self.beta = torch.linspace(1e-4, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Enhanced UNet backbone
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Encoder
        self.enc1 = nn.ModuleList([
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        ])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Decoder with time embedding
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.ModuleList([
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # +64 for time embedding
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        ])

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.view(-1, 64, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Encoder
        e1 = x
        for layer in self.enc1:
            e1 = layer(e1)
        p1 = self.pool1(e1)
        
        # Decoder with time embedding
        up1 = self.up1(p1)
        concat1 = torch.cat([up1, e1, t_emb], dim=1)
        out = concat1
        for layer in self.dec1:
            out = layer(out)
        return out

    def diffusion_step(self, x_t, t):
        # Get noise schedule for current step
        alpha_t = self.alpha_bar[t]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        # Add noise according to diffusion schedule
        noise = torch.randn_like(x_t)
        x_noisy = torch.sqrt(alpha_t) * x_t + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise

def add_noise(image, noise_factor=0.1):
    noise = torch.randn_like(image) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

# Training setup
def train_model(data_dir, num_epochs=100):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = DenoiseDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = DiffusionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with diffusion
    for epoch in range(num_epochs):
        for batch_idx, (_, clean_images) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, model.n_steps, (clean_images.shape[0],))
            
            # Apply forward diffusion
            x_noisy, noise = model.diffusion_step(clean_images, t)
            
            # Predict noise
            predicted_noise = model(x_noisy, t.float())
            
            # Calculate loss
            loss = criterion(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
    
    return model

@torch.no_grad()
def denoise_sample(model, noisy_image, n_steps=100):
    x = noisy_image
    for t in reversed(range(n_steps)):
        t_tensor = torch.tensor([t]).float()
        predicted_noise = model(x, t_tensor)
        alpha_t = model.alpha_bar[t]
        alpha_prev = model.alpha_bar[t-1] if t > 0 else torch.tensor(1.)
        
        # Reverse diffusion step
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t)) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            x = x + sigma_t * noise
    
    return x

if __name__ == "__main__":
    data_dir = "path/to/your/data"  # Update this path
    model = train_model(data_dir)
    # Save the model
    torch.save(model.state_dict(), "denoising_model.pth")
