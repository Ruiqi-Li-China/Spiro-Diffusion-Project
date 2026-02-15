import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.vq_vae import VQVAE
import os

# Configuration
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_latents():
    print(f"--- Generating Latent Representations on {DEVICE} ---")
    
    # 1. Load Data
    data_path = 'data/processed/signals_L512.npy'
    if not os.path.exists(data_path):
        print("Error: Signal file not found.")
        return

    # Load and reshape signals for the model [Batch, 1, 512]
    signals = np.load(data_path).astype(np.float32)
    signals = signals[:, np.newaxis, :] 
    
    dataset = TensorDataset(torch.from_numpy(signals))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) # Shuffle=False to keep order matching metadata
    
    # 2. Load Trained VQ-VAE
    model = VQVAE(num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(DEVICE)
    
    checkpoint_path = 'checkpoints/vqvae_phase1.pth'
    if not os.path.exists(checkpoint_path):
        print("Error: VQ-VAE checkpoint not found! Train it first.")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # 3. Encoding Loop
    all_latents = []
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(DEVICE)
            
            # Pass through Encoder
            z = model._encoder(data)
            z = model._pre_vq_conv(z)
            
            # We want the quantized indices (discrete tokens) or the continuous latents
            # For Latent Diffusion, we typically use the continuous latent vectors 'z'
            all_latents.append(z.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
    
    # 4. Save
    all_latents = np.concatenate(all_latents, axis=0)
    save_path = 'data/processed/latents.npy'
    np.save(save_path, all_latents)
    
    print(f"--- Success! ---")
    print(f"Original Shape: {signals.shape} (Raw Signals)")
    print(f"Latent Shape:   {all_latents.shape} (Compressed Representation)")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    generate_latents()
