import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.vq_vae import VQVAE
import os

# Configuration based on Proposal [cite: 73-75]
BATCH_SIZE = 128 # increased batch size for speed
NUM_EPOCHS = 20 # usually enough for initial convergence
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"--- Starting VQ-VAE Training on {DEVICE} ---")
    
    # 1. Load the Multimodal Data
    data_path = 'data/processed/signals_L512.npy'
    if not os.path.exists(data_path):
        print("Error: processed data not found!")
        return
        
    signals = np.load(data_path).astype(np.float32)
    # Reshape for PyTorch 1D Conv: [Batch, Channels=1, Length=512]
    signals = signals[:, np.newaxis, :] 
    
    # Create DataLoader
    dataset = TensorDataset(torch.from_numpy(signals))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    model = VQVAE(num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_recon_error = 0
        total_vq_loss = 0
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            vq_loss, data_recon, _ = model(data)
            
            # Reconstruction Loss (MSE)
            recon_error = torch.mean((data_recon - data)**2)
            
            # Total Loss = Recon + VQ Loss [cite: 70]
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            
            total_recon_error += recon_error.item()
            total_vq_loss += vq_loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{batch_idx+1}/{len(dataloader)}] "
                      f"Recon Error: {recon_error.item():.4f}")
        
        avg_recon = total_recon_error / len(dataloader)
        print(f"=== Epoch {epoch+1} Completed. Avg Recon Error: {avg_recon:.5f} ===")
        
    # 4. Save the Model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vqvae_phase1.pth')
    print("--- Model Saved to checkpoints/vqvae_phase1.pth ---")

if __name__ == "__main__":
    train()
