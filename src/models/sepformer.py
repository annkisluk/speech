
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

        x = self.conv(x)
        x = torch.relu(x)
        return x


class SimplifiedDecoder(nn.Module):

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

        x = self.conv_transpose(x)
        
        if target_length is not None:
            if x.shape[-1] > target_length:
                x = x[..., :target_length]
            elif x.shape[-1] < target_length:
                padding = target_length - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, padding))
        
        return x


class SimplifiedMaskingNetwork(nn.Module):

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
        #Build simplified model for development
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

        # Encoder - Use standard PyTorch Conv1d since SpeechBrain Conv1d has different API
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=n_basis,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size // 2
        )
        
        # Masking Network (Dual-Path Transformer) 
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
        
        # Decoder 
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

        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        
        with torch.no_grad():
            encoded = self.encoder(noisy)
        
        return encoded
    
    def freeze_encoder(self):

        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen (φ_E^0)")
    
    def freeze_masking_network(self):

        if self.masking_network is not None:
            for param in self.masking_network.parameters():
                param.requires_grad = False
            print("Masking network frozen (θ^0)")
        else:
            print("Masking network not present (using adapter layers instead)")
    
    def freeze_backbone(self):

        self.freeze_encoder()
        self.freeze_masking_network()
        print("Backbone frozen (encoder + masking network)")
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count parameters in model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


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


