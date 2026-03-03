
import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math


class Adapter(nn.Module):


    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 1,
        activation: str = "relu",
        init_scale: float = 0.01
    ):

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

        nn.init.normal_(self.down_project.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.normal_(self.up_project.weight, mean=0.0, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.down_project(x)   # [B, L, C] -> [B, L, Ĉ]
        h = self.activation(h)
        output = self.up_project(h) # [B, L, Ĉ] -> [B, L, C]
        return output

    def get_num_parameters(self) -> int:
        """Return number of trainable parameters in this adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterWithSelector(nn.Module):


    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 1,
        max_adapters: int = 10,
        activation: str = "relu",
        init_scale: float = 0.01
    ):

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

        if adapter_idx >= self.num_adapters:
            raise ValueError(f"Adapter {adapter_idx} does not exist")

        self.betas.zero_()
        self.betas[adapter_idx] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        adapter_idx: Optional[int] = None
    ) -> torch.Tensor:

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

        if adapter_idx >= self.num_adapters:
            raise ValueError(f"Adapter {adapter_idx} does not exist")

        for param in self.adapters[adapter_idx].parameters():
            param.requires_grad = False

    def unfreeze_adapter(self, adapter_idx: int):

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_type = "FFL"


class MHAAdapter(AdapterWithSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_type = "MHA"


class TransformerBlockWithAdapters(nn.Module):
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

        # MHA sub-layer with parallel adapter (Paper Eq. 1)
        # Both MHA and adapter receive the same normalized input h_m
        residual = x
        h = self.norm1(x)
        attn_out, _ = self.mha(h, h, h)
        attn_out = self.dropout1(attn_out)

        if self.use_mha_adapter:
            attn_out = attn_out + self.mha_adapter(h, mha_adapter_idx)

        x = residual + attn_out

        # FFN sub-layer with parallel adapter 
        # f_m = A^t_ffl(h_m) + FFN(h_m)
        residual = x
        h = self.norm2(x)
        ffn_out = self.ffn(h)

        if self.use_ffl_adapter:
            ffn_out = ffn_out + self.ffl_adapter(h, ffl_adapter_idx)

        x = residual + ffn_out

        return x

    def add_new_session_adapters(self, bottleneck_dim: int = 1) -> Dict[str, int]:

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
