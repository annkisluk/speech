
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .sepformer import SepFormer
from .adapters import TransformerBlockWithAdapters, FFLAdapter, MHAAdapter


class DualPathBlock(nn.Module):

    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        bottleneck_dim: int = 1,
        max_sessions: int = 10,
        layers_per_direction: int = 8,
        use_mha_adapter: bool = True,
        use_ffl_adapter: bool = True,
        skip_around_intra: bool = True,
    ):
        super().__init__()
        
        self.skip_around_intra = skip_around_intra
        self.layers_per_direction = layers_per_direction
        
        # Intra-chunk transformer layers (local attention within each chunk)
        self.intra_layers = nn.ModuleList([
            TransformerBlockWithAdapters(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                bottleneck_dim=bottleneck_dim,
                use_mha_adapter=use_mha_adapter,
                use_ffl_adapter=use_ffl_adapter,
                max_adapters=max_sessions
            )
            for _ in range(layers_per_direction)
        ])
        self.intra_linear = nn.Linear(d_model, d_model)
        self.intra_norm = nn.GroupNorm(1, d_model)
        
        # Inter-chunk transformer layers (global attention across chunks)
        self.inter_layers = nn.ModuleList([
            TransformerBlockWithAdapters(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                bottleneck_dim=bottleneck_dim,
                use_mha_adapter=use_mha_adapter,
                use_ffl_adapter=use_ffl_adapter,
                max_adapters=max_sessions
            )
            for _ in range(layers_per_direction)
        ])
        self.inter_linear = nn.Linear(d_model, d_model)
        self.inter_norm = nn.GroupNorm(1, d_model)
    
    def forward(self, x: torch.Tensor, session_id: int = 0) -> torch.Tensor:

        B, N, K, S = x.shape
        
        #Intra-chunk processing (local context within each chunk)
        intra = x.permute(0, 3, 2, 1).contiguous().reshape(B * S, K, N)
        
        adapter_kwargs = {}
        if session_id > 0:
            idx = session_id - 1
            adapter_kwargs = {'mha_adapter_idx': idx, 'ffl_adapter_idx': idx}
        
        for layer in self.intra_layers:
            intra = layer(intra, **adapter_kwargs)
        
        intra = self.intra_linear(intra)  # [B*S, K, N]
        intra = intra.reshape(B, S, K, N).permute(0, 3, 2, 1).contiguous()  # [B, N, K, S]
        
        if self.skip_around_intra:
            intra = intra + x  # Skip connection around intra
        intra = self.intra_norm(intra)
        
        # Inter-chunk processing (global context across chunks)
        inter = intra.permute(0, 2, 3, 1).contiguous().reshape(B * K, S, N)
        
        for layer in self.inter_layers:
            inter = layer(inter, **adapter_kwargs)
        
        inter = self.inter_linear(inter)  # [B*K, S, N]
        inter = inter.reshape(B, K, S, N).permute(0, 3, 1, 2).contiguous()  # [B, N, K, S]
        inter = inter + intra  # Always skip for inter
        inter = self.inter_norm(inter)
        
        return inter
    
    def add_new_session_adapters(self, bottleneck_dim: int = 1):
        #Add adapter pair to all intra + inter layers for a new session
        indices = []
        for layer in self.intra_layers:
            indices.append(layer.add_new_session_adapters(bottleneck_dim))
        for layer in self.inter_layers:
            indices.append(layer.add_new_session_adapters(bottleneck_dim))
        return indices
    
    def freeze_session_adapters(self, adapter_idx: int):
        #Freeze a session's adapters in all layers
        for layer in self.intra_layers:
            layer.freeze_session_adapters(adapter_idx)
        for layer in self.inter_layers:
            layer.freeze_session_adapters(adapter_idx)
    
    def set_active_adapters(self, adapter_idx: int):
        #Set active adapter in all layers
        for layer in self.intra_layers:
            layer.set_active_adapters(adapter_idx)
        for layer in self.inter_layers:
            layer.set_active_adapters(adapter_idx)
    
    def get_all_transformer_layers(self) -> List[TransformerBlockWithAdapters]:
        #Return flat list of all TransformerBlockWithAdapters
        return list(self.intra_layers) + list(self.inter_layers)


class LNAModel(nn.Module):
   
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
        chunk_size: int = 250,
        num_blocks: int = 2
    ):
        super().__init__()
        
        self.n_basis = n_basis
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.max_sessions = max_sessions
        self.use_mha_adapter = use_mha_adapter
        self.use_ffl_adapter = use_ffl_adapter
        self.chunk_size = chunk_size
        
        # SepFormer backbone - only use encoder and decoder
        self.sepformer = SepFormer(
            n_basis=n_basis,
            kernel_size=kernel_size,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_speechbrain=False
        )
        
        # Delete unused masking_network (we replace it with DPT blocks)
        del self.sepformer.masking_network
        self.sepformer.masking_network = None
        
        # Input normalization before dual-path processing
        # SpeechBrain: GroupNorm(1, N) — equivalent to channel-wise LayerNorm
        self.input_norm = nn.GroupNorm(1, n_basis)
        
        # Dual-Path Transformer Blocks
        # 2 DPT blocks, each with 8 intra + 8 inter layers
        # Total: 2 × (8 + 8) = 32 transformer layers
        self.dpt_blocks = nn.ModuleList([
            DualPathBlock(
                d_model=n_basis,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bottleneck_dim=adapter_bottleneck_dim,
                max_sessions=max_sessions,
                layers_per_direction=num_layers,
                use_mha_adapter=use_mha_adapter,
                use_ffl_adapter=use_ffl_adapter,
                skip_around_intra=True
            )
            for _ in range(num_blocks)
        ])
        
        # Mask estimation output:
        # SpeechBrain: PReLU → Conv1d → ReLU
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
        if session_id == 0:
            raise ValueError("Session 0 is pre-training, use pretrain mode")
        
        if session_id > self.max_sessions:
            raise ValueError(f"Session {session_id} exceeds max sessions {self.max_sessions}")
        
        if bottleneck_dim is None:
            bottleneck_dim = self.adapter_bottleneck_dim
        
        device = next(self.parameters()).device
        
        # Add adapters to all layers in all DPT blocks
        adapter_indices = []
        for block in self.dpt_blocks:
            indices = block.add_new_session_adapters(bottleneck_dim)
            adapter_indices.extend(indices)
            block.to(device)
        
        # Add new decoder (no padding, no bias — matching SepFormer)
        decoder_key = f'session_{session_id}'
        self.decoders[decoder_key] = nn.ConvTranspose1d(
            in_channels=self.n_basis,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.kernel_size // 2,
            bias=False
        ).to(device)
        
        # Initialize from pretrained decoder so the new decoder starts
        # from a known-good solution and only needs to fine-tune
        pretrained_dec = self.decoders['session_0']
        src = pretrained_dec.conv_transpose if hasattr(pretrained_dec, 'conv_transpose') else pretrained_dec
        with torch.no_grad():
            self.decoders[decoder_key].weight.copy_(src.weight)
            if src.bias is not None and self.decoders[decoder_key].bias is not None:
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
            
            # Check if we need to add adapters (check first layer of first block)
            first_layer = self.dpt_blocks[0].intra_layers[0]
            if first_layer.mha_adapter.num_adapters <= adapter_idx:
                print(f"Adding new adapters for session {session_id} (adapter_idx={adapter_idx})")
                for block in self.dpt_blocks:
                    block.add_new_session_adapters(bottleneck_dim=self.adapter_bottleneck_dim)
                    block.to(device)
            
            # Add decoder for this session if needed
            decoder_key = f'session_{session_id}'
            if decoder_key not in self.decoders:
                self.decoders[decoder_key] = nn.ConvTranspose1d(
                    in_channels=self.n_basis,
                    out_channels=1,
                    kernel_size=self.kernel_size,
                    stride=self.kernel_size // 2,
                    bias=False
                ).to(device)
                # Initialize from pretrained decoder
                pretrained_dec = self.decoders['session_0']
                src = pretrained_dec.conv_transpose if hasattr(pretrained_dec, 'conv_transpose') else pretrained_dec
                with torch.no_grad():
                    self.decoders[decoder_key].weight.copy_(src.weight)
                    if src.bias is not None and self.decoders[decoder_key].bias is not None:
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
                
                # Freeze DPT block components: transformer layers, linears, norms
                for block in self.dpt_blocks:
                    for name, param in block.named_parameters():
                        # Only freeze non-adapter parameters
                        if 'mha_adapter' not in name and 'ffl_adapter' not in name:
                            param.requires_grad = False
            
            # Freeze previous adapters (adapter 0 = session 1, adapter 1 = session 2, ...)
            if freeze_previous_adapters:
                for block in self.dpt_blocks:
                    for prev_adapter_idx in range(session_id - 1):
                        block.freeze_session_adapters(prev_adapter_idx)
            
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
            for block in self.dpt_blocks:
                block.set_active_adapters(session_id - 1)  # -1 because adapters are 0-indexed
            
            print(f"Training mode: Session {session_id} (incremental)")
            print(f"  Trainable parameters: {self.get_num_parameters(trainable_only=True):,}")
    
    def set_inference_mode(self, session_id: int):

        self.eval()
        self.current_session = session_id
        
        # Set active adapters
        if session_id > 0:
            for block in self.dpt_blocks:
                block.set_active_adapters(session_id - 1)
        
        print(f"Inference mode: Session {session_id}")
    
    def _segment(self, x, chunk_size):

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
        
        # 3. Dual-path transformer processing via DPT blocks
        x = segments  # [B, N, K, S]
        
        for block in self.dpt_blocks:
            x = block(x, session_id)
        
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

        return self.sepformer.get_encoder_output(noisy)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        #Count parameters in model
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_adapter_info(self) -> Dict:
        #Get information about all adapters
        adapter_params = 0
        total_transformer_layers = 0
        for block in self.dpt_blocks:
            for layer in block.get_all_transformer_layers():
                total_transformer_layers += 1
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
            'num_dpt_blocks': len(self.dpt_blocks),
            'total_transformer_layers': total_transformer_layers,
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

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.current_session = checkpoint['session_id']
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Session: {checkpoint['session_id']}")
        print(f"  Adapter info: {checkpoint.get('adapter_info', {})}")
        
        return checkpoint
