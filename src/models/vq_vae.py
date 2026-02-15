import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Implements the Codebook logic from Proposal Section 3.1.2
    Discretizes the continuous latent vectors into k defined prototypes.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        # The Codebook C = {c1, c2, ... cK} [cite: 63]
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

    def forward(self, inputs):
        # inputs shape: [Batch, Channel, Length] -> [Batch, Length, Channel] for calculation
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances: ||z - e_j||^2 [cite: 66]
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding: z = argmin ||Encoder(x) - c_z||
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss: Reconstruction Loss + Commitment Loss [cite: 70]
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator (copy gradients back)
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 2, 1).contiguous(), encoding_indices

class VQVAE(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_hiddens=32, 
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # 1D Encoder (Compresses L=512 -> L=64) [cite: 65]
        self._encoder = nn.Sequential(
            nn.Conv1d(1, num_hiddens // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_hiddens // 2, num_hiddens, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_hiddens, num_hiddens, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_hiddens, embedding_dim, 1, stride=1)
        )
        
        self._pre_vq_conv = nn.Conv1d(embedding_dim, embedding_dim, 1)
        
        # Vector Quantization Layer
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # 1D Decoder (Reconstructs L=64 -> L=512)
        self._decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, num_hiddens, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(num_hiddens, num_hiddens // 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(num_hiddens // 2, 1, 4, stride=2, padding=1)
        )

    def forward(self, x):
        # Encode
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        
        # Quantize
        loss, quantized, perplexity = self._vq_vae(z)
        
        # Decode
        x_recon = self._decoder(quantized)
        
        return loss, x_recon, perplexity
