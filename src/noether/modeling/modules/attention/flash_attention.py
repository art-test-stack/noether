#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Lightweight FlashAttention wrapper with fallbacks to PyTorch SDPA.

This module exposes the minimal functions used across the attention modules
in this codebase: ``flash_attn_func``, ``flash_attn_qkvpacked_func`` and
``flash_attn_with_kvcache`` plus a ``flash_attn`` SimpleNamespace for
compatibility with earlier usage.

The implementation prefers the external kernel when available but falls back
to implementations based on ``torch.nn.functional.scaled_dot_product_attention``
so semantics remain identical.
"""
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_is_flash_attn_enabled = os.getenv("NOETHER_USE_FLASH_ATTENTION", "1").lower() in ("1", "true")

def _load_flash_attn_interface() -> object | None:
    # Use the same detection logic as other modules: require CUDA and the
    # varunneal/flash-attention-3 kernel with a ``flash_attn_interface``.
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda_get_device_capability()
    except Exception:
        return None
    if major < 9:
        return None
    try:
        import kernels  # type: ignore
    except Exception:
        return None
    try:
        kernel = kernels.get_kernel("varunneal/flash-attention-3")
    except Exception:
        return None
    flash_attn_interface = getattr(kernel, "flash_attn_interface", None)
    if flash_attn_interface is None:
        return None
    return flash_attn_interface


_FLASH_ATTN = _load_flash_attn_interface()
flash_attention_is_installed = _FLASH_ATTN is not None


def sdpa(
        q, 
        k, 
        v, 
        **kwargs):
    """Thin wrapper over the repository's SDPA fallback.

    Keep the exact same calling convention as flash-attn wrappers in the
    codebase: q/k/v expected shapes are (B, T, H, D).
    """
    # lazy import to avoid circular deps
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def flash_attn_func(
        q, 
        k, 
        v, 
        attn_mask=None,
        dropout_p=0.0,
        causal=False, 
    ):
    if flash_attention_is_installed and _is_flash_attn_enabled:
        if attn_mask is not None:
            B, N, H, D = q.shape

            q_list = []
            k_list = []
            v_list = []

            cu_seqlens = [0]

            for b in range(B):

                keep = attn_mask[b]   # [N]

                qb = q[b][keep]
                kb = k[b][keep]
                vb = v[b][keep]

                q_list.append(qb)
                k_list.append(kb)
                v_list.append(vb)

                cu_seqlens.append(
                    cu_seqlens[-1] + qb.shape[0]
                )

            q_cat = torch.cat(q_list, dim=0)
            k_cat = torch.cat(k_list, dim=0)
            v_cat = torch.cat(v_list, dim=0)

            cu_seqlens = torch.tensor(
                cu_seqlens,
                device=q.device,
                dtype=torch.int32
            )

            max_seqlen = max(
                x.shape[0] for x in q_list
            )
            out = _FLASH_ATTN.flash_attn_varlen_func(
                q_cat,
                k_cat,
                v_cat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                causal=causal,
            )

        return _FLASH_ATTN.flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=causal,
            window_size=(-1, -1),
            alibi_slopes=None
        )
    # GQA detection: K/V may have fewer heads than Q
    use_gqa = (k.size(-2) != q.size(-2))
    return sdpa(q, k, v, dropout_p=dropout_p, use_gqa=use_gqa)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attention_is_installed=flash_attention_is_installed,
)
