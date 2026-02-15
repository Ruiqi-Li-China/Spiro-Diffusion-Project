import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes the diffusion time step 't' into a vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ClinicalEmbedding(nn.Module):
    """
    Encodes Age, Height, Gender into a 'Context Vector' for Cross-Attention.
    Ref: Proposal Innovation Point 1 (Multimodal Guidance)
    """
    def __init__(self, context_dim=64):
        super().__init__()
        # Inputs: Age (1), Height (1), Gender (1) -> Total 3
        self.project = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, context_dim),
            nn.SiLU()
        )
    
    def forward(self, x):
        # x shape: [Batch, 3]
        return self.project(x).unsqueeze(1) # [Batch, 1, Context_Dim]

class ResidualBlock(nn.Module):
    """Standard ResNet block for 1D signals."""
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # Projection for Time Embeddings
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time embedding (broadcast over length)
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class ConditionalUNet1D(nn.Module):
    """
    The main cLDM Backbone.
    Input: Noisy Latent (x_t) + Time (t) + Clinical Data (c)
    Output: Predicted Noise (epsilon)
    """
    def __init__(self, in_channels=64, context_dim=64):
        super().__init__()
        self.in_channels = in_channels
        
        # Time Embedding
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Clinical Embedding (The "Guidance")
        self.clinical_encoder = ClinicalEmbedding(context_dim)

        # Downsampling (Encoder)
        self.down1 = ResidualBlock(in_channels, 128, time_dim)
        self.down2 = ResidualBlock(128, 256, time_dim)
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck (Cross-Attention could go here, simplified to concatenation for 1D)
        self.mid1 = ResidualBlock(256, 256, time_dim)
        
        # Upsampling (Decoder)
        self.up1 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.up_block1 = ResidualBlock(128 + 128, 128, time_dim) # Skip connection
        
        self.out = nn.Conv1d(128, in_channels, 3, padding=1)
        
        # Fusion of Clinical Context (Simple Cross-Attention via concatenation/projection)
        self.context_fusion = nn.Linear(context_dim, time_dim)

    def forward(self, x, t, clinical_data):
        # 1. Embeddings
        t_emb = self.time_mlp(t)
        
        # Fuse Clinical Data into Time Embedding (Global Conditioning)
        # This tells the model: "Generate a curve for a 45yo Male, 175cm"
        c_emb = self.clinical_encoder(clinical_data).squeeze(1)
        t_emb = t_emb + self.context_fusion(c_emb)
        
        # 2. U-Net Path
        # Down
        x1 = self.down1(x, t_emb) # [B, 128, L]
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb) # [B, 256, L/2]
        
        # Mid
        mid = self.mid1(x2, t_emb)
        
        # Up
        up = self.up1(mid) # [B, 128, L]
        # Skip Connection
        up = torch.cat([up, x1], dim=1) 
        out = self.up_block1(up, t_emb)
        
        return self.out(out)
