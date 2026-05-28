#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn

from noether.core.schemas.modules.attention import AttentionConfig
from noether.modeling.functional.init import apply_init_method
from noether.modeling.functional.rope import rope


def _load_flash_attn3_interface() -> Any | None:
    if not torch.cuda.is_available():
        return None

    try:
        major, _ = torch.cuda.get_device_capability()
    except RuntimeError:
        return None

    if major < 9:
        return None

    try:
        import kernels
    except ImportError:
        return None

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        kernel = kernels.get_kernel("varunneal/flash-attention-3")
    except Exception:
        return None

    flash_attn_interface = getattr(kernel, "flash_attn_interface", None)
    if flash_attn_interface is None:
        return None
    if not hasattr(flash_attn_interface, "flash_attn_func"):
        return None

    return flash_attn_interface


_FLASH_ATTN3 = _load_flash_attn3_interface()


class DotProductAttentionConfig(AttentionConfig):
    """Configuration for the Dot Product attention module."""



class DotProductAttention(nn.Module):
    """Scaled dot-product attention module."""

    def __init__(
        self,
        config: AttentionConfig,
    ):
        """

        Args:
            config: Configuration for the DotProductAttention module. See
                :class:`~noether.core.schemas.modules.AttentionConfig` for available options.
        """

        super().__init__()

        config = DotProductAttentionConfig(**config.model_dump())

        if not (config.hidden_dim % config.num_heads == 0):
            raise ValueError("The 'dim' must be divisible by 'num_heads'.")

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.init_weights = config.init_weights
        self.use_rope = config.use_rope
        self.use_flash_attn = config.use_flash_attn
        self.dropout = config.dropout
        self.proj_dropout = nn.Dropout(config.dropout)

        self.q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        if config.qk_norm:
            self.q_norm: nn.Module = nn.RMSNorm(self.head_dim)
            self.k_norm: nn.Module = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        apply_init_method(self, self.proj.weight, self.init_weights)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
            attn_mask: For causal attention (i.e., no attention over the future token) a attention mask should be provided. Defaults to None.
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys. None if use_rope=False.

        Returns:
            Returns the output of the attention module.
        """

        qkv_weight = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        qkv_bias = torch.cat([self.q.bias, self.k.bias, self.v.bias], dim=0) if self.q.bias is not None else None
        qkv = F.linear(x, qkv_weight, qkv_bias)

        q, k, v = einops.rearrange(
            qkv,
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            assert freqs is not None
            q = rope(q, freqs=freqs)
            k = rope(k, freqs=freqs)
        else:
            assert freqs is None

        if self.use_flash_attn and attn_mask is None and _FLASH_ATTN3 is not None:
            x = _FLASH_ATTN3.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            ).transpose(1, 2)
        else:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
            )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj_dropout(self.proj(x))

        return x
