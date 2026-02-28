"""
SepFormer Backbone for Speech Enhancement

This module wraps SpeechBrain's SepFormer implementation to serve as
the backbone for the LNA system.

Paper Reference: Section III.A, Reference [17]

Reference [17]: "Attention is All You Need in Speech Separation"
Subakan et al., ICASSP 2021

SepFormer Architecture:
1. Encoder: 1D Conv → learns latent representation
2. Masking Network: Transformer-based (our focus for adapters)
3. Decoder: Transposed Conv → reconstructs waveform

Paper: "We adopt Sepformer as the backbone" (Section III.B)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import warnings

try:
    from speechbrain.lobes.models.dual_path import Dual_Path_Model
    from speechbrain.nnet.CNN import Conv1d
    from speechbrain.nnet.linear import Linear
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    warnings.warn(
        "SpeechBrain not installed. Install with: pip install speechbrain\n"
        "Using simplified SepFormer implementation for development."
    )


class SimplifiedEncoder(nn.Module):
    """
    SepFormer Encoder matching SpeechBrain's implementation.
    
    Architecture: Conv1d (no bias, no padding) → ReLU
    Reference: SpeechBrain speechbrain/lobes/models/dual_path.py Encoder class
    """
    
    def __init__(
        self,
        kernel_size: int = 16,
        out_channels: int = 256
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T] audio waveform
        Returns:
            [B, N, L] encoded representation where L = floor((T - kernel_size) / stride) + 1
        """
        x = self.conv(x)
        x = torch.relu(x)
        return x


class SimplifiedDecoder(nn.Module):
    """
    SepFormer Decoder matching SpeechBrain's implementation.
    
    Architecture: ConvTranspose1d (no bias, no padding)
    Reference: SpeechBrain speechbrain/lobes/models/dual_path.py Decoder class
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        kernel_size: int = 16
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )
    
    def forward(self, x: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, L] encoded representation
            target_length: Target output length (for trimming)
        Returns:
            [B, 1, T] reconstructed waveform
        """
        x = self.conv_transpose(x)
        
        if target_length is not None:
            if x.shape[-1] > target_length:
                x = x[..., :target_length]
            elif x.shape[-1] < target_length:
                padding = target_length - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, padding))
        
        return x


class SimplifiedMaskingNetwork(nn.Module):
    """
    Simplified masking network for development/testing
    
    Real SepFormer masking network:
    - Dual-path transformer architecture
    - Processes local and global context
    - Predicts mask for enhancement
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output projection to get mask
        self.output_proj = nn.Linear(in_channels, in_channels)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, L] encoded features
        Returns:
            [B, N, L] mask
        """
        # Transpose for transformer: [B, L, N]
        x = x.transpose(1, 2)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Project and get mask
        mask = self.output_proj(x)
        mask = self.output_activation(mask)
        
        # Transpose back: [B, N, L]
        mask = mask.transpose(1, 2)
        
        return mask


class SepFormer(nn.Module):
    """
    SepFormer model for speech enhancement
    
    Paper Reference: Section III.B
    "We adopt Sepformer as the backbone, where the network parameters 
    are kept the same as those in [17]."
    
    Architecture (from Reference [17]):
        Input waveform [B, 1, T]
            ↓
        Encoder: Conv1D → [B, N, L]
            ↓
        Masking Network: Transformer → [B, N, L]
            ↓
        Mask × Encoded → [B, N, L]
            ↓
        Decoder: ConvTranspose1D → [B, 1, T]
            ↓
        Enhanced waveform [B, 1, T]
    
    Parameters from paper Section IV.A:
    - N = 256 (number of basis signals)
    - L = 16 (length of basis signals)
    - Transformer: 8 layers, 8 heads
    """
    
    def __init__(
        self,
        n_basis: int = 256,
        kernel_size: int = 16,
        num_layers: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_speechbrain: bool = True
    ):
        """
        Args:
            n_basis: Number of basis signals (N in paper)
            kernel_size: Kernel size for encoder/decoder (L in paper)
            num_layers: Number of transformer layers
            nhead: Number of attention heads
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            use_speechbrain: Whether to use SpeechBrain's implementation
        """
        super().__init__()
        
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.use_speechbrain = use_speechbrain and SPEECHBRAIN_AVAILABLE
        
        if self.use_speechbrain:
            print("Using SpeechBrain SepFormer implementation")
            self._build_speechbrain_model(
                n_basis, kernel_size, num_layers, 
                nhead, dim_feedforward, dropout
            )
        else:
            print("Using simplified SepFormer implementation")
            self._build_simplified_model(
                n_basis, kernel_size, num_layers,
                nhead, dim_feedforward, dropout
            )
    
    def _build_simplified_model(
        self,
        n_basis: int,
        kernel_size: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        """Build simplified model for development"""
        self.encoder = SimplifiedEncoder(kernel_size, n_basis)
        self.masking_network = SimplifiedMaskingNetwork(
            n_basis, num_layers, nhead, dim_feedforward, dropout
        )
        self.decoder = SimplifiedDecoder(n_basis, kernel_size)
    
    def _build_speechbrain_model(
        self,
        n_basis: int,
        kernel_size: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float
    ):
        """
        Build model using SpeechBrain's Dual-Path architecture
        
        This is the full SepFormer from the paper.
        """
        # Encoder - Use standard PyTorch Conv1d since SpeechBrain Conv1d has different API
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=n_basis,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2
        )
        
        # Masking Network (Dual-Path Transformer) - This is the key component from paper
        self.masking_network = Dual_Path_Model(
            in_channels=n_basis,
            out_channels=n_basis,
            intra_model='transformer',
            inter_model='transformer',
            num_layers=num_layers,
            norm='ln',
            K=kernel_size,
            num_spks=1,  # Single speaker for enhancement
            skip_around_intra=True,
            linear_layer_after_inter_intra=False
        )
        
        # Decoder - Use standard PyTorch
        self.decoder = nn.ConvTranspose1d(
            in_channels=n_basis,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2
        )
    
    def forward(
        self,
        noisy: torch.Tensor,
        return_mask: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through SepFormer
        
        Args:
            noisy: Noisy input waveform [B, 1, T] or [B, T]
            return_mask: If True, return (enhanced, mask)
        
        Returns:
            enhanced: Enhanced waveform [B, 1, T]
            (optional) mask: Estimated mask [B, N, L]
        """
        # Ensure input is [B, 1, T]
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        
        batch_size, _, input_length = noisy.shape
        
        # Encode: [B, 1, T] → [B, N, L]
        encoded = self.encoder(noisy)
        
        # Estimate mask: [B, N, L] → [B, N, L]
        mask = self.masking_network(encoded)
        
        # Apply mask
        masked = encoded * mask
        
        # Decode: [B, N, L] → [B, 1, T]
        enhanced = self.decoder(masked, target_length=input_length)
        
        if return_mask:
            return enhanced, mask
        return enhanced
    
    def get_encoder_output(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Get encoder output for noise selector
        
        Paper Section III.D:
        "We use the feature extractor E(·; φ_E^0) of the pre-trained model"
        
        Args:
            noisy: Noisy input [B, 1, T]
        
        Returns:
            Encoded features [B, N, L]
        """
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        
        with torch.no_grad():
            encoded = self.encoder(noisy)
        
        return encoded
    
    def freeze_encoder(self):
        """
        Freeze encoder parameters
        
        Paper: "the parameters of the encoder, φ_E^0, remain unchanged"
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen (φ_E^0)")
    
    def freeze_masking_network(self):
        """
        Freeze masking network parameters
        
        Paper: "Masking network: θ^0 (frozen after pre-training)"
        """
        if self.masking_network is not None:
            for param in self.masking_network.parameters():
                param.requires_grad = False
            print("Masking network frozen (θ^0)")
        else:
            print("Masking network not present (using adapter layers instead)")
    
    def freeze_backbone(self):
        """
        Freeze encoder and masking network (for incremental learning)
        
        Paper: "we employ a frozen pre-trained model to train and retain 
        a domain-specific adapter for each newly encountered domain"
        """
        self.freeze_encoder()
        self.freeze_masking_network()
        print("Backbone frozen (encoder + masking network)")
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count parameters in model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Factory Function
# ============================================================================

def create_sepformer(
    n_basis: int = 256,
    kernel_size: int = 16,
    num_layers: int = 8,
    nhead: int = 8,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None
) -> SepFormer:
    """
    Factory function to create SepFormer model
    
    Args:
        n_basis: Number of basis signals (paper: 256)
        kernel_size: Encoder/decoder kernel size (paper: 16)
        num_layers: Number of transformer layers (paper: 8)
        nhead: Number of attention heads (paper: 8)
        dim_feedforward: FFN dimension (paper: 1024)
        dropout: Dropout rate (paper: 0.1)
        pretrained: Load pre-trained weights
        checkpoint_path: Path to checkpoint file
    
    Returns:
        SepFormer model
    """
    model = SepFormer(
        n_basis=n_basis,
        kernel_size=kernel_size,
        num_layers=num_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    if pretrained and checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing SepFormer model...")
    
    # Create model
    print("\n1. Creating SepFormer:")
    model = SepFormer(
        n_basis=256,
        kernel_size=16,
        num_layers=2,  # Small for testing
        nhead=8,
        use_speechbrain=False  # Use simplified for testing
    )
    
    print(f"   Total parameters: {model.get_num_parameters():,}")
    print(f"   Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass:")
    batch_size = 2
    audio_length = 16000  # 2 seconds at 8kHz
    
    noisy = torch.randn(batch_size, 1, audio_length)
    enhanced = model(noisy)
    
    print(f"   Input shape: {noisy.shape}")
    print(f"   Output shape: {enhanced.shape}")
    assert enhanced.shape == noisy.shape, "Output shape mismatch!"
    
    # Test encoder output
    print("\n3. Testing encoder output (for noise selector):")
    encoded = model.get_encoder_output(noisy)
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   [B, N, L] = [{encoded.shape[0]}, {encoded.shape[1]}, {encoded.shape[2]}]")
    
    # Test freezing
    print("\n4. Testing parameter freezing:")
    print(f"   Before freezing: {model.get_num_parameters(trainable_only=True):,} trainable")
    model.freeze_backbone()
    print(f"   After freezing: {model.get_num_parameters(trainable_only=True):,} trainable")
    
    print("\n✓ SepFormer model working!")