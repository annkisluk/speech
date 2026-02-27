"""
Adapter Modules for Incremental Speech Enhancement
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math


class Adapter(nn.Module):
    """
    Base adapter module: down-project -> ReLU -> up-project.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 1,
        activation: str = "relu",
        init_scale: float = 0.01
    ):
        """
        Args:
            input_dim: Input and output dimension C
            bottleneck_dim: Compressed dimension Ĉ (paper uses 1)
            activation: Non-linearity between projections
            init_scale: Small init scale so adapter starts near zero output
        """
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Down-projection: C -> Ĉ (W*,d in paper)
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Up-projection: Ĉ -> C (W*,u in paper)
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=False)

        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        """
        Initialize weights near zero so the adapter starts with minimal effect
        on the frozen model's output, allowing stable training from the start.
        """
        nn.init.normal_(self.down_project.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, mean=0.0, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, C]

        Returns:
            Adapter output [B, L, C] - a domain-specific correction signal
        """
        h = self.down_project(x)   # [B, L, C] -> [B, L, Ĉ]
        h = self.activation(h)
        output = self.up_project(h) # [B, L, Ĉ] -> [B, L, C]
        return output

    def get_num_parameters(self) -> int:
        """Return number of trainable parameters in this adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterWithSelector(nn.Module):
    """
    Manages a list of adapters (one per session) with beta gating.

    Betas are binary switches set manually, not learned:
    - During training session t: beta_t=1, all others=0
    - During inference: beta_j=1 for domain j predicted by noise selector

    Paper Equation (3): g_m = beta1*e1_m + beta2*e2_m + ... + betat*e^t_m
    Paper Equation (6): beta^l = 1 if l == j, else 0
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 1,
        max_adapters: int = 10,
        activation: str = "relu",
        init_scale: float = 0.01
    ):
        """
        Args:
            input_dim: Input dimension
            bottleneck_dim: Adapter bottleneck dimension
            max_adapters: Maximum number of sessions supported
            activation: Activation function
            init_scale: Initialization scale
        """
        super().__init__()

        self.input_dim = input_dim
        self.max_adapters = max_adapters

        # Grows by one adapter per session
        self.adapters = nn.ModuleList()

        # Beta values are not learned - set manually via set_active_adapter()
        self.register_buffer('betas', torch.zeros(max_adapters))

        self.num_adapters = 0

    def add_adapter(
        self,
        bottleneck_dim: int = 1,
        activation: str = "relu",
        init_scale: float = 0.01
    ) -> int:
        """
        Create and register a new adapter for a new session.

        Returns:
            Index of the newly created adapter
        """
        if self.num_adapters >= self.max_adapters:
            raise ValueError(f"Maximum number of adapters ({self.max_adapters}) reached")

        adapter = Adapter(
            input_dim=self.input_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            init_scale=init_scale
        )

        self.adapters.append(adapter)
        adapter_idx = self.num_adapters
        self.num_adapters += 1

        return adapter_idx

    def set_active_adapter(self, adapter_idx: int):
        """
        Activate one adapter and deactivate all others by setting betas.

        Args:
            adapter_idx: Index of the adapter to activate
        """
        if adapter_idx >= self.num_adapters:
            raise ValueError(f"Adapter {adapter_idx} does not exist")

        self.betas.zero_()
        self.betas[adapter_idx] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        adapter_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute weighted sum of adapter outputs using beta values.

        Args:
            x: Input tensor [B, L, C]
            adapter_idx: If given, bypass betas and use this adapter directly

        Returns:
            Combined adapter output [B, L, C]
        """
        # No adapters yet (pre-training phase) - contribute nothing
        if self.num_adapters == 0:
            return torch.zeros_like(x)

        # Direct selection bypasses beta mechanism (used during training)
        if adapter_idx is not None:
            if adapter_idx >= self.num_adapters:
                raise ValueError(f"Adapter {adapter_idx} does not exist")
            return self.adapters[adapter_idx](x)

        # Beta-weighted sum: in practice only one beta is 1, rest are 0
        output = torch.zeros_like(x)
        for i in range(self.num_adapters):
            if self.betas[i] > 0:
                output = output + self.betas[i] * self.adapters[i](x)

        return output

    def freeze_adapter(self, adapter_idx: int):
        """
        Freeze a trained adapter so it cannot be modified in future sessions.

        Args:
            adapter_idx: Index of adapter to freeze
        """
        if adapter_idx >= self.num_adapters:
            raise ValueError(f"Adapter {adapter_idx} does not exist")

        for param in self.adapters[adapter_idx].parameters():
            param.requires_grad = False

    def unfreeze_adapter(self, adapter_idx: int):
        """
        Unfreeze an adapter to make it trainable again.

        Args:
            adapter_idx: Index of adapter to unfreeze
        """
        if adapter_idx >= self.num_adapters:
            raise ValueError(f"Adapter {adapter_idx} does not exist")

        for param in self.adapters[adapter_idx].parameters():
            param.requires_grad = True

    def get_adapter_info(self) -> Dict:
        """Return summary of current adapter state."""
        return {
            'num_adapters': self.num_adapters,
            'max_adapters': self.max_adapters,
            'active_adapters': [i for i in range(self.num_adapters) if self.betas[i] > 0],
            'total_parameters': sum(a.get_num_parameters() for a in self.adapters)
        }


class FFLAdapter(AdapterWithSelector):
    """
    Feed-Forward Layer Adapter (FFL-A).
    Inserted after the FFN sub-layer in each transformer block.
    Architecturally identical to MHAAdapter - named separately for clarity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_type = "FFL"


class MHAAdapter(AdapterWithSelector):
    """
    Multi-Head Attention Adapter (MHA-A).
    Inserted after the MHA sub-layer in each transformer block.
    Architecturally identical to FFLAdapter - named separately for clarity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_type = "MHA"


class TransformerBlockWithAdapters(nn.Module):
    """
    Standard transformer block (MHA + FFN) with FFL-A and MHA-A inserted.

    The frozen pre-trained weights are unchanged. Each adapter adds a small
    domain-specific correction in parallel with its respective sub-layer.

    Signal flow (Paper Equations 1-2):
        x -> LayerNorm -> h
             h -> MHA -----------> sum --> residual add -> x'
             h -> MHA-A(adapter) -/
        x' -> LayerNorm -> h'
             h' -> FFN -----------> sum --> residual add -> output
             h' -> FFL-A(adapter) -/
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        bottleneck_dim: int = 1,
        use_mha_adapter: bool = True,
        use_ffl_adapter: bool = True,
        max_adapters: int = 10
    ):
        """
        Args:
            d_model: Model dimension C
            nhead: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout probability
            bottleneck_dim: Adapter bottleneck Ĉ
            use_mha_adapter: Whether to include MHA-A
            use_ffl_adapter: Whether to include FFL-A
            max_adapters: Maximum sessions supported
        """
        super().__init__()

        self.use_mha_adapter = use_mha_adapter
        self.use_ffl_adapter = use_ffl_adapter

        # Multi-Head Attention sub-layer
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # MHA adapter sits after the attention output
        if use_mha_adapter:
            self.mha_adapter = MHAAdapter(
                input_dim=d_model,
                bottleneck_dim=bottleneck_dim,
                max_adapters=max_adapters
            )

        # Feed-Forward sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFL adapter sits after the FFN output
        if use_ffl_adapter:
            self.ffl_adapter = FFLAdapter(
                input_dim=d_model,
                bottleneck_dim=bottleneck_dim,
                max_adapters=max_adapters
            )

    def forward(
        self,
        x: torch.Tensor,
        mha_adapter_idx: Optional[int] = None,
        ffl_adapter_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            mha_adapter_idx: Which MHA adapter to use (None = use betas)
            ffl_adapter_idx: Which FFL adapter to use (None = use betas)

        Returns:
            Output tensor [B, L, D]
        """
        # MHA sub-layer with parallel adapter (Paper Eq. 1)
        # Both MHA and adapter receive the same normalized input h_m
        residual = x
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h)
        attn_out = self.dropout1(attn_out)

        if self.use_mha_adapter:
            attn_out = attn_out + self.mha_adapter(h, mha_adapter_idx)

        x = residual + attn_out

        # FFN sub-layer with parallel adapter (Paper Eq. 2)
        # f_m = A^t_ffl(h_m) + FFN(h_m)
        residual = x
        h = self.norm2(x)
        ffn_out = self.ffn(h)

        if self.use_ffl_adapter:
            ffn_out = ffn_out + self.ffl_adapter(h, ffl_adapter_idx)

        x = residual + ffn_out

        return x

    def add_new_session_adapters(self, bottleneck_dim: int = 1) -> Dict[str, int]:
        """
        Add a new MHA and FFL adapter pair for an incoming session.

        Returns:
            Dict with indices of newly created adapters
        """
        indices = {}

        if self.use_mha_adapter:
            indices['mha'] = self.mha_adapter.add_adapter(bottleneck_dim)

        if self.use_ffl_adapter:
            indices['ffl'] = self.ffl_adapter.add_adapter(bottleneck_dim)

        return indices

    def set_active_adapters(self, adapter_idx: int):
        """Activate a specific session's adapters for inference."""
        if self.use_mha_adapter:
            self.mha_adapter.set_active_adapter(adapter_idx)
        if self.use_ffl_adapter:
            self.ffl_adapter.set_active_adapter(adapter_idx)

    def freeze_session_adapters(self, adapter_idx: int):
        """Freeze a specific session's adapters so they cannot be modified."""
        if self.use_mha_adapter:
            self.mha_adapter.freeze_adapter(adapter_idx)
        if self.use_ffl_adapter:
            self.ffl_adapter.freeze_adapter(adapter_idx)


def count_adapter_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters belonging to adapters vs total model parameters.
    Used to verify the paper's claim that adapters use fewer than 2% of parameters.

    Args:
        model: Model containing adapters

    Returns:
        Dict with adapter_parameters, total_parameters, adapter_percentage
    """
    adapter_params = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (Adapter, AdapterWithSelector)):
            adapter_params += sum(p.numel() for p in module.parameters())
        total_params += sum(p.numel() for p in module.parameters())

    return {
        'adapter_parameters': adapter_params,
        'total_parameters': total_params,
        'adapter_percentage': 100 * adapter_params / max(total_params, 1)
    }


if __name__ == "__main__":
    print("Testing Adapter modules...")

    adapter = Adapter(input_dim=256, bottleneck_dim=1)
    x = torch.randn(2, 100, 256)
    out = adapter(x)
    print(f"Adapter - Input: {x.shape}, Output: {out.shape}, Params: {adapter.get_num_parameters()}")

    adapter_selector = AdapterWithSelector(input_dim=256, bottleneck_dim=1, max_adapters=5)
    for i in range(3):
        idx = adapter_selector.add_adapter()
        print(f"Added adapter {idx}")

    adapter_selector.set_active_adapter(1)
    out = adapter_selector(x)
    print(f"AdapterWithSelector output: {out.shape}")
    print(f"Adapter info: {adapter_selector.get_adapter_info()}")

    block = TransformerBlockWithAdapters(d_model=256, nhead=8, bottleneck_dim=1)
    indices = block.add_new_session_adapters(bottleneck_dim=1)
    print(f"New session adapter indices: {indices}")
    block.set_active_adapters(0)
    out = block(x)
    print(f"Block output: {out.shape}")

    param_info = count_adapter_parameters(block)
    print(f"Adapter params: {param_info['adapter_parameters']}, "
          f"Total: {param_info['total_parameters']}, "
          f"Percentage: {param_info['adapter_percentage']:.2f}%")

    print("All adapter modules working!")