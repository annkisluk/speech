"""
Complete LNA (Learning Noise Adapters) Model

This module integrates:
1. SepFormer backbone (frozen after pre-training)
2. Domain-specific adapters (FFL-A and MHA-A)
3. Session-specific decoders
4. Noise selector for inference

Paper Reference: Section III - The Proposed Method

Key Innovation:
"We introduce a lightweight ISE module, referred to as Learning Noise 
Adapters (LNAs). When faced with new noise domains, LNAs dynamically 
train noise adapters tailored to adapt to the specific domain, while 
maintaining the stability of pre-trained modules."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .sepformer import SepFormer
from .adapters import TransformerBlockWithAdapters, FFLAdapter, MHAAdapter


class LNAModel(nn.Module):
    """
    Complete Learning Noise Adapters Model
    
    Paper Reference: Figure 2 - Framework of LNA
    
    Architecture:
        Session 0 (Pre-training):
            Input → Encoder → Dual-Path Masking Network → Decoder → Output
        
        Session t (Incremental, t>0):
            Input → Encoder(frozen) → Dual-Path Masking Network(frozen) 
                  → + Adapters^t → Decoder^t → Output
    
    Uses Dual-Path processing (SepFormer-style):
        - Segments encoded features into overlapping chunks
        - Alternates Intra-chunk (local) and Inter-chunk (global) transformer layers
        - Each transformer layer has optional adapters (MHA-A and FFL-A)
        - Generates a mask applied to encoded features
    """
    
    def __init__(
        self,
        n_basis: int = 256,
        kernel_size: int = 16,
        num_layers: int = 8,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        adapter_bottleneck_dim: int = 1,
        max_sessions: int = 10,
        use_mha_adapter: bool = True,
        use_ffl_adapter: bool = True,
        chunk_size: int = 250
    ):
        """
        Args:
            n_basis: Number of basis signals in SepFormer
            kernel_size: Encoder/decoder kernel size
            num_layers: Number of transformer layers (split between intra/inter)
            nhead: Number of attention heads
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            adapter_bottleneck_dim: Bottleneck dim for adapters (Ĉ in paper)
            max_sessions: Maximum number of incremental sessions
            use_mha_adapter: Whether to use MHA adapters
            use_ffl_adapter: Whether to use FFL adapters
            chunk_size: Chunk size K for dual-path segmentation
        """
        super().__init__()
        
        self.n_basis = n_basis
        self.num_layers = num_layers
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.max_sessions = max_sessions
        self.use_mha_adapter = use_mha_adapter
        self.use_ffl_adapter = use_ffl_adapter
        self.chunk_size = chunk_size
        
        # SepFormer backbone - only use encoder and decoder
        # Paper: "We adopt Sepformer as the backbone"
        self.sepformer = SepFormer(
            n_basis=n_basis,
            kernel_size=kernel_size,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_speechbrain=False
        )
        
        # Delete unused masking_network (we replace it with dual-path adapter layers)
        del self.sepformer.masking_network
        self.sepformer.masking_network = None
        
        # Input normalization for the masking network
        # SepFormer: LayerNorm before dual-path processing
        self.input_norm = nn.GroupNorm(1, n_basis)
        
        # Dual-Path Transformer Layers with Adapters
        # Paper: SepFormer uses intra-chunk and inter-chunk transformer layers
        # We alternate: even layers = intra-chunk, odd layers = inter-chunk
        self.adapter_layers = nn.ModuleList()
        for _ in range(num_layers):
            adapter_layer = TransformerBlockWithAdapters(
                d_model=n_basis,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bottleneck_dim=adapter_bottleneck_dim,
                use_mha_adapter=use_mha_adapter,
                use_ffl_adapter=use_ffl_adapter,
                max_adapters=max_sessions
            )
            self.adapter_layers.append(adapter_layer)
        
        # SepFormer mask estimation output:
        # PReLU activation + Linear projection → mask
        # Paper/SepFormer: After dual-path, project to mask with ReLU
        self.mask_prelu = nn.PReLU()
        self.mask_proj = nn.Conv1d(n_basis, n_basis, 1)
        self.mask_activation = nn.ReLU()
        
        # Session-specific decoders
        self.decoders = nn.ModuleDict()
        self.decoders['session_0'] = self.sepformer.decoder
        
        # Track current session
        self.current_session = 0
        self.is_pretrained = False
    
    def add_new_session(
        self,
        session_id: int,
        bottleneck_dim: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Add adapters and decoder for a new incremental session
        
        Paper: "When faced with new noise domains, LNAs dynamically train 
        noise adapters tailored to adapt to the specific domain"
        
        Args:
            session_id: ID of new session (1, 2, 3, ...)
            bottleneck_dim: Override default bottleneck dimension
        
        Returns:
            Dictionary with information about added components
        """
        if session_id == 0:
            raise ValueError("Session 0 is pre-training, use pretrain mode")
        
        if session_id > self.max_sessions:
            raise ValueError(f"Session {session_id} exceeds max sessions {self.max_sessions}")
        
        if bottleneck_dim is None:
            bottleneck_dim = self.adapter_bottleneck_dim
        
        device = next(self.parameters()).device
        
        # Add adapters to all layers
        adapter_indices = []
        for layer in self.adapter_layers:
            indices = layer.add_new_session_adapters(bottleneck_dim)
            adapter_indices.append(indices)
            layer.to(device)
        
        # Add new decoder
        # Paper: "create new decoder φ^t_D for session t"
        # Must match session 0 decoder architecture (kernel=16, stride=8, padding=8)
        decoder_key = f'session_{session_id}'
        self.decoders[decoder_key] = nn.ConvTranspose1d(
            in_channels=self.n_basis,
            out_channels=1,
            kernel_size=16,
            stride=8,
            padding=8
        ).to(device)
        
        # Initialize from pretrained decoder so the new decoder starts
        # from a known-good solution and only needs to fine-tune
        pretrained_dec = self.decoders['session_0']
        src = pretrained_dec.conv_transpose if hasattr(pretrained_dec, 'conv_transpose') else pretrained_dec
        with torch.no_grad():
            self.decoders[decoder_key].weight.copy_(src.weight)
            self.decoders[decoder_key].bias.copy_(src.bias)
        
        self.current_session = session_id
        
        info = {
            'session_id': session_id,
            'bottleneck_dim': bottleneck_dim,
            'num_adapter_layers': len(adapter_indices),
            'decoder_key': decoder_key
        }
        
        print(f"Added session {session_id}: {info}")
        return info
    
    def set_training_mode(
        self,
        session_id: int,
        freeze_backbone: bool = True,
        freeze_previous_adapters: bool = True,
        freeze_previous_decoders: bool = True
    ):
        """
        Set training mode for a specific session
        
        Paper: Section III - Incremental Learning Configuration
        
        For session 0 (pre-training):
            - Train everything
        
        For session t > 0 (incremental):
            - Freeze: encoder (φ_E^0), masking network (θ^0)
            - Freeze: previous adapters and decoders
            - Train: new adapters (A^t_f, A^t_m) and decoder (φ^t_D)
        
        Args:
            session_id: Current training session
            freeze_backbone: Freeze encoder + masking network (paper: always True for t>0)
            freeze_previous_adapters: Freeze adapters from previous sessions
            freeze_previous_decoders: Freeze decoders from previous sessions
        """
        self.current_session = session_id
        
        if session_id == 0:
            # Pre-training: Train everything
            self.train()
            self.is_pretrained = False
            print("Training mode: Session 0 (pre-training) - training all parameters")
            
        else:
            # Incremental learning
            self.train()
            
            # Determine the device of the existing model
            device = next(self.parameters()).device
            
            # Add new adapters for this session if needed
            adapter_idx = session_id - 1  # Session 1 uses adapter 0, etc.
            # Check if we need to add adapters (only for the first layer)
            if self.adapter_layers[0].mha_adapter.num_adapters <= adapter_idx:
                print(f"Adding new adapters for session {session_id} (adapter_idx={adapter_idx})")
                for layer in self.adapter_layers:
                    layer.add_new_session_adapters(bottleneck_dim=self.adapter_bottleneck_dim)
                    # Move new adapter params to correct device
                    layer.to(device)
            
            # Add decoder for this session if needed
            decoder_key = f'session_{session_id}'
            if decoder_key not in self.decoders:
                self.decoders[decoder_key] = nn.ConvTranspose1d(
                    in_channels=self.n_basis,
                    out_channels=1,
                    kernel_size=16,
                    stride=8,
                    padding=8
                ).to(device)
                # Initialize from pretrained decoder
                pretrained_dec = self.decoders['session_0']
                src = pretrained_dec.conv_transpose if hasattr(pretrained_dec, 'conv_transpose') else pretrained_dec
                with torch.no_grad():
                    self.decoders[decoder_key].weight.copy_(src.weight)
                    self.decoders[decoder_key].bias.copy_(src.bias)
            
            # Freeze backbone (encoder + masking network)
            if freeze_backbone:
                self.sepformer.freeze_backbone()
                
                # Freeze the pre-trained masking components (part of θ^0)
                for param in self.input_norm.parameters():
                    param.requires_grad = False
                for param in self.mask_prelu.parameters():
                    param.requires_grad = False
                for param in self.mask_proj.parameters():
                    param.requires_grad = False
                # mask_activation (ReLU) has no parameters
                
                # Freeze base transformer layers (MHA, FFN, norms) - part of θ^0
                for layer in self.adapter_layers:
                    for name, param in layer.named_parameters():
                        # Only freeze non-adapter parameters
                        if 'mha_adapter' not in name and 'ffl_adapter' not in name:
                            param.requires_grad = False
            
            # Freeze previous adapters (adapter 0 = session 1, adapter 1 = session 2, ...)
            if freeze_previous_adapters:
                for layer in self.adapter_layers:
                    for prev_adapter_idx in range(session_id - 1):
                        layer.freeze_session_adapters(prev_adapter_idx)
            
            # Freeze previous decoders
            if freeze_previous_decoders:
                for prev_session in range(session_id):
                    prev_key = f'session_{prev_session}'
                    if prev_key in self.decoders:
                        for param in self.decoders[prev_key].parameters():
                            param.requires_grad = False
            
            # Unfreeze current session's decoder
            if decoder_key in self.decoders:
                for param in self.decoders[decoder_key].parameters():
                    param.requires_grad = True
            
            # Set active adapters
            for layer in self.adapter_layers:
                layer.set_active_adapters(session_id - 1)  # -1 because adapters are 0-indexed
            
            print(f"Training mode: Session {session_id} (incremental)")
            print(f"  Trainable parameters: {self.get_num_parameters(trainable_only=True):,}")
    
    def set_inference_mode(self, session_id: int):
        """
        Set inference mode for a specific session
        
        Args:
            session_id: Session to use for inference
        """
        self.eval()
        self.current_session = session_id
        
        # Set active adapters
        if session_id > 0:
            for layer in self.adapter_layers:
                layer.set_active_adapters(session_id - 1)
        
        print(f"Inference mode: Session {session_id}")
    
    def _segment(self, x, chunk_size):
        """Segment encoded features into overlapping chunks (50% overlap).
        
        Paper: SepFormer uses dual-path segmentation.
        
        Args:
            x: [B, N, L] encoded features
            chunk_size: K, the chunk length
        
        Returns:
            segments: [B, N, K, S] chunked features
            original_length: L for later reconstruction
        """
        B, N, L = x.shape
        hop = chunk_size // 2  # 50% overlap
        
        # Pad L so it can be evenly segmented
        rest = L % hop
        if rest > 0:
            pad_len = hop - rest
            x = F.pad(x, (0, pad_len))
        
        L_padded = x.shape[2]
        
        # Ensure at least one full chunk
        if L_padded < chunk_size:
            x = F.pad(x, (0, chunk_size - L_padded))
            L_padded = chunk_size
        
        # Create overlapping segments using unfold
        segments = x.unfold(2, chunk_size, hop)  # [B, N, S, K]
        segments = segments.permute(0, 1, 3, 2).contiguous()  # [B, N, K, S]
        
        return segments, L
    
    def _overlap_add(self, segments, chunk_size, original_length):
        """Reconstruct signal from overlapping segments via overlap-add.
        
        Uses fold (col2im) for proper vectorized overlap-add, avoiding
        in-place ops that cause CUDA misaligned address errors with DataParallel.
        
        Args:
            segments: [B, N, K, S]
            chunk_size: K
            original_length: L before padding
        
        Returns:
            output: [B, N, L]
        """
        B, N, K, S = segments.shape
        hop = chunk_size // 2
        out_len = (S - 1) * hop + chunk_size
        
        # Reshape for fold: [B, N*K, S]
        segments_flat = segments.reshape(B, N * K, S)
        
        # fold performs overlap-add: [B, N*K, S] → [B, N, out_len]
        output = F.fold(
            segments_flat,
            output_size=(N, out_len),
            kernel_size=(N, K),
            stride=(N, hop)
        ).squeeze(1)  # [B, 1, N, out_len] → [B, N, out_len]
        
        # Compute overlap count for normalization
        ones = torch.ones_like(segments_flat)
        count = F.fold(
            ones,
            output_size=(N, out_len),
            kernel_size=(N, K),
            stride=(N, hop)
        ).squeeze(1)
        
        output = output / count.clamp(min=1)
        
        return output[:, :, :original_length]

    def forward(
        self,
        noisy: torch.Tensor,
        session_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through LNA model with Dual-Path processing.
        
        Paper: SepFormer backbone with intra-chunk and inter-chunk attention.
        
        Flow:
            1. Encode noisy waveform
            2. Segment into overlapping chunks
            3. Alternate intra-chunk (local) and inter-chunk (global) transformer layers
            4. Generate mask from transformer output
            5. Apply mask to encoded features
            6. Overlap-add to reconstruct
            7. Decode with session-specific decoder
        
        Args:
            noisy: Noisy input waveform [B, 1, T] or [B, T]
            session_id: Which session's decoder to use (None = current_session)
        
        Returns:
            Enhanced waveform [B, 1, T]
        """
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        
        if session_id is None:
            session_id = self.current_session
        
        batch_size, _, input_length = noisy.shape
        
        # 1. Encode
        encoded = self.sepformer.encoder(noisy)  # [B, N, L]
        B, N, L = encoded.shape
        
        # Apply input normalization
        x = self.input_norm(encoded)  # [B, N, L]
        
        # 2. Segment into overlapping chunks for dual-path processing
        K = self.chunk_size
        segments, orig_L = self._segment(x, K)  # [B, N, K, S]
        S = segments.shape[3]
        
        # 3. Dual-path transformer processing
        # Alternate between intra-chunk (even layers) and inter-chunk (odd layers)
        x = segments  # [B, N, K, S]
        
        for i, layer in enumerate(self.adapter_layers):
            if i % 2 == 0:
                # --- Intra-chunk (local context within each chunk) ---
                # Reshape: [B, N, K, S] → [B*S, K, N]
                x_proc = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
                
                if session_id > 0:
                    idx = session_id - 1
                    x_proc = layer(x_proc, mha_adapter_idx=idx, ffl_adapter_idx=idx)
                else:
                    x_proc = layer(x_proc, mha_adapter_idx=None, ffl_adapter_idx=None)
                
                # Reshape back: [B*S, K, N] → [B, N, K, S]
                x = x_proc.view(B, S, K, N).permute(0, 3, 2, 1).contiguous()
            else:
                # --- Inter-chunk (global context across chunks) ---
                # Reshape: [B, N, K, S] → [B*K, S, N]
                x_proc = x.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
                
                if session_id > 0:
                    idx = session_id - 1
                    x_proc = layer(x_proc, mha_adapter_idx=idx, ffl_adapter_idx=idx)
                else:
                    x_proc = layer(x_proc, mha_adapter_idx=None, ffl_adapter_idx=None)
                
                # Reshape back: [B*K, S, N] → [B, N, K, S]
                x = x_proc.view(B, K, S, N).permute(0, 3, 1, 2).contiguous()
        
        # 4. Overlap-add to reconstruct: [B, N, K, S] → [B, N, L]
        x = self._overlap_add(x, K, orig_L)
        
        # 5. Generate and apply mask
        # SepFormer: PReLU → Conv1d → ReLU → multiply with encoded features
        x = self.mask_prelu(x)
        mask = self.mask_proj(x)        # [B, N, L]
        mask = self.mask_activation(mask)  # ReLU — non-negative mask
        masked = encoded * mask           # Apply mask to encoded features
        
        # 6. Decode using session-specific decoder
        decoder_key = f'session_{session_id}'
        if decoder_key in self.decoders:
            decoder = self.decoders[decoder_key]
        else:
            decoder = self.decoders['session_0']
        
        enhanced = decoder(masked)
        
        # Ensure correct output length
        if enhanced.shape[-1] > input_length:
            enhanced = enhanced[..., :input_length]
        elif enhanced.shape[-1] < input_length:
            padding = input_length - enhanced.shape[-1]
            enhanced = F.pad(enhanced, (0, padding))
        
        return enhanced
    
    def get_encoder_features(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Extract features from encoder for noise selector
        
        Paper Section III.D:
        "We use the feature extractor E(·; φ_E^0) of the pre-trained model 
        to initialize the domain selector"
        
        Args:
            noisy: Noisy input [B, 1, T]
        
        Returns:
            Encoded features [B, N, L]
        """
        return self.sepformer.get_encoder_output(noisy)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Count parameters in model"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_adapter_info(self) -> Dict:
        """Get information about all adapters"""
        adapter_params = 0
        for layer in self.adapter_layers:
            if self.use_mha_adapter:
                adapter_params += sum(
                    p.numel() for p in layer.mha_adapter.parameters()
                )
            if self.use_ffl_adapter:
                adapter_params += sum(
                    p.numel() for p in layer.ffl_adapter.parameters()
                )
        
        total_params = self.get_num_parameters()
        
        return {
            'current_session': self.current_session,
            'num_adapter_layers': len(self.adapter_layers),
            'adapter_parameters': adapter_params,
            'total_parameters': total_params,
            'adapter_percentage': 100 * adapter_params / max(total_params, 1),
            'decoders': list(self.decoders.keys())
        }
    
    def save_checkpoint(
        self,
        path: str,
        session_id: int,
        optimizer_state: Optional[Dict] = None,
        **kwargs
    ):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            session_id: Current session ID
            optimizer_state: Optimizer state dict
            **kwargs: Additional metadata
        """
        checkpoint = {
            'session_id': session_id,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'n_basis': self.n_basis,
                'adapter_bottleneck_dim': self.adapter_bottleneck_dim,
                'max_sessions': self.max_sessions,
                'use_mha_adapter': self.use_mha_adapter,
                'use_ffl_adapter': self.use_ffl_adapter
            },
            'adapter_info': self.get_adapter_info()
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        checkpoint.update(kwargs)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False,
        strict: bool = True
    ) -> Dict:
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to return optimizer state
            strict: Whether to strictly enforce state_dict keys match
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.current_session = checkpoint['session_id']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Session: {checkpoint['session_id']}")
        print(f"  Adapter info: {checkpoint.get('adapter_info', {})}")
        
        return checkpoint


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing LNA Model...")
    
    # Create model
    print("\n1. Creating LNA Model:")
    model = LNAModel(
        n_basis=256,
        num_layers=2,  # Small for testing
        nhead=8,
        adapter_bottleneck_dim=1,
        max_sessions=5
    )
    
    print(f"   Total parameters: {model.get_num_parameters():,}")
    
    # Test pre-training mode (Session 0)
    print("\n2. Testing Session 0 (Pre-training):")
    model.set_training_mode(session_id=0)
    
    noisy = torch.randn(2, 1, 16000)
    enhanced = model(noisy, session_id=0)
    print(f"   Input: {noisy.shape}, Output: {enhanced.shape}")
    
    # Add incremental session
    print("\n3. Adding Session 1 (Incremental):")
    model.add_new_session(session_id=1, bottleneck_dim=1)
    model.set_training_mode(session_id=1)
    
    enhanced = model(noisy, session_id=1)
    print(f"   Session 1 output: {enhanced.shape}")
    
    # Add more sessions
    print("\n4. Adding Sessions 2-3:")
    for session_id in [2, 3]:
        model.add_new_session(session_id=session_id)
        print(f"   Added session {session_id}")
    
    # Check adapter info
    print("\n5. Adapter Information:")
    info = model.get_adapter_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Test checkpoint save/load
    print("\n6. Testing checkpoint:")
    model.save_checkpoint(
        "checkpoints/test_lna.pt",
        session_id=3,
        test_metric=0.95
    )
    
    print("\n✓ LNA Model working!")