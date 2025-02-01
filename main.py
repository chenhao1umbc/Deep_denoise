import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
from PIL import Image
import os
import numpy as np

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
            image = Image.open(file_path).convert('RGB')
            image = np.array(image) / 255.0
        else:
            # Handle .avi files using imageio
            reader = iio.imiter(file_path)
            image = next(reader)  # Get first frame
            image = image / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        # Add noise to create noisy-clean pairs
        noisy_image = add_noise(image)
        return noisy_image, image

# Simplified UNet-style model for diffusion
class DiffusionModel(nn.Module):
    def __init__(self, n_steps=1000):
        super(DiffusionModel, self).__init__()
        self.n_steps = n_steps
        
        # Define beta schedule
        self.beta = torch.linspace(1e-4, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),  # +64 for time embedding
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.view(-1, 64, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Encoder
        e1 = self.enc1(x)
        
        # Decoder with time embedding
        concat1 = torch.cat([e1, t_emb], dim=1)
        return self.dec1(concat1)

    def diffusion_step(self, x_t, t):
        # Get noise schedule for current step
        alpha_t = self.alpha_bar[t]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        # Add noise according to diffusion schedule
        noise = torch.randn_like(x_t)
        x_noisy = torch.sqrt(alpha_t) * x_t + torch.sqrt(1 - alpha_t) * noise
        return x_noisy, noise

def add_noise(image, noise_factor=0.1):
    if isinstance(image, torch.Tensor):
        noise = torch.randn_like(image) * noise_factor
    else:
        noise = torch.randn_like(torch.from_numpy(image)) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)

def preprocess(image):
    # Resize and convert to tensor
    image = image.resize((256, 256), Image.BILINEAR)
    image = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
    return image

def train_model(data_dir, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    dataset = DenoiseDataset(data_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = DiffusionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with diffusion
    for epoch in range(num_epochs):
        for batch_idx, (_, clean_images) in enumerate(dataloader):
            clean_images = clean_images.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, model.n_steps, (clean_images.shape[0],)).to(device)
            
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
    device = next(model.parameters()).device
    x = noisy_image.to(device)
    
    for t in reversed(range(n_steps)):
        t_tensor = torch.tensor([t], device=device).float()
        predicted_noise = model(x, t_tensor)
        alpha_t = model.alpha_bar[t].to(device)
        alpha_prev = model.alpha_bar[t-1].to(device) if t > 0 else torch.tensor(1.).to(device)
        
        # Reverse diffusion step
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t)) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_prev)
            x = x + sigma_t * noise
    
    return x.cpu()

if __name__ == "__main__":
    data_dir = "path/to/your/data"  # Update this path
    model = train_model(data_dir)
    # Save the model
    torch.save(model.state_dict(), "denoising_model.pth")
