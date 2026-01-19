#!/usr/bin/env python3
"""
Joan Kusuma's +Layer Fashion Embedding Model
Exact implementation from her research with mAP@5 = 0.53
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class JoanKusumaEmbeddingModel(nn.Module):
    """
    Joan Kusuma's +Layer Model Architecture
    - 8 convolutional layers + 4 max-pooling layers in encoder
    - Reduces spatial dimensions to 512×1×1 (512-dimensional embedding)
    - 8 transposed convolutional layers in decoder
    - Achieves mAP@5 = 0.53 on fashion dataset
    """
    def __init__(self):
        super(JoanKusumaEmbeddingModel, self).__init__()
        
        # Encoder: 8 conv layers + 4 max-pooling layers
        # Input: 3×128×128 → Output: 512×1×1
        self.encoder = nn.Sequential(
            # Block 1: 128×128 → 64×64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 128×128 → 64×64
            
            # Block 2: 64×64 → 32×32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 64×64 → 32×32
            
            # Block 3: 32×32 → 16×16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # 32×32 → 16×16
            
            # Block 4: 16×16 → 1×1 (+Layer improvement)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(16, stride=16),  # 16×16 → 1×1 (Global pooling effect)
        )
        
        # Decoder: 8 transposed conv layers (matching encoder complexity)
        # Input: 512×1×1 → Output: 3×128×128
        self.decoder = nn.Sequential(
            # Upsample from 1×1 to 16×16
            nn.ConvTranspose2d(512, 512, kernel_size=16, stride=16, padding=0),
            nn.ReLU(True),
            
            # Block 4 decode: 16×16 → 16×16
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            # Block 3 decode: 16×16 → 32×32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            # Block 2 decode: 32×32 → 64×64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            
            # Block 1 decode: 64×64 → 128×128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output activation for image reconstruction
        )
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Extract 512-dimensional embeddings"""
        with torch.no_grad():
            encoded = self.encoder(x)
            # Flatten to 512-dimensional vector
            return encoded.view(encoded.size(0), -1)

# Legacy support - use same interface as original model
class FeaturizerModel(JoanKusumaEmbeddingModel):
    """Alias for backward compatibility"""
    pass