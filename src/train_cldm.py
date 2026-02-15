import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.diffusion_unet import ConditionalUNet1D
import os

# --- Configuration ---
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTEPS = 1000

def train_diffusion():
    print(f"--- Starting Robust cLDM Training on {DEVICE} ---")
    
    # 1. Load Data
    latents = np.load('data/processed/latents.npy').astype(np.float32)
    meta_df = pd.read_csv('data/processed/metadata_aligned.csv')
    
    print(f"Original Dataset Size: {len(latents)}")

    # 2. Robust Data Cleaning (The Fix for your 220 missing heights)
    # Create a mask that is True only if Age, Height, AND Gender are valid (not NaN)
    valid_indices = ~meta_df[['age', 'height', 'gender']].isnull().any(axis=1)
    
    # Apply the mask to both Latents and Metadata to keep them aligned
    latents = latents[valid_indices]
    meta_df = meta_df[valid_indices]
    
    # Reset index to ensure clean access
    meta_df.reset_index(drop=True, inplace=True)
    
    print(f"Cleaned Dataset Size: {len(latents)} (Removed {~valid_indices.sum()} rows with missing data)")
    
    # 3. Normalize Metadata (0-1 range helps stability)
    meta_features = meta_df[['age', 'height', 'gender']].values.astype(np.float32)
    meta_features[:, 0] = meta_features[:, 0] / 100.0   # Scale Age
    meta_features[:, 1] = meta_features[:, 1] / 200.0   # Scale Height
    meta_features[:, 2] = meta_features[:, 2] - 1.0     # Map Gender (1,2) -> (0,1)
    
    # Create Dataset
    dataset = TensorDataset(torch.from_numpy(latents), torch.from_numpy(meta_features))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Initialize Model
    model = ConditionalUNet1D(in_channels=64).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()
    
    # Pre-calculate Noise Schedule (Beta Schedule)
    beta = torch.linspace(0.0001, 0.02, TIMESTEPS).to(DEVICE)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    
    # 5. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        
        for batch_idx, (x_0, clinical) in enumerate(dataloader):
            x_0 = x_0.to(DEVICE)
            clinical = clinical.to(DEVICE)
            
            # Sample random time steps
            t = torch.randint(0, TIMESTEPS, (x_0.shape[0],), device=DEVICE).long()
            
            # Add Noise
            noise = torch.randn_like(x_0)
            a_hat_t = alpha_hat[t][:, None, None] # Reshape for broadcasting
            x_t = torch.sqrt(a_hat_t) * x_0 + torch.sqrt(1 - a_hat_t) * noise
            
            # Predict Noise
            predicted_noise = model(x_t, t, clinical)
            
            # Calculate Loss
            loss = mse(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Prevents explosion if a bad value sneaks in)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f} ===")
        
    # 6. Save Model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cldm_phase2.pth')
    print("--- Success! cLDM Saved to checkpoints/cldm_phase2.pth ---")

if __name__ == "__main__":
    train_diffusion()
