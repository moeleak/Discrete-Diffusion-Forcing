from functools import lru_cache
import math

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    compute_in_float32: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if compute_in_float32:
        working = x.to(torch.float32)
        x1, x2 = torch.chunk(working, 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)
    else:
        working = x
        cos = torch.cat((cos, cos), dim=-1).to(x.dtype)
        sin = torch.cat((sin, sin), dim=-1).to(x.dtype)
    x1, x2 = torch.chunk(working, 2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (working * cos + rotated * sin).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        compute_in_float32: bool = True,
        inv_freq: torch.Tensor | None = None,
        attention_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.compute_in_float32 = compute_in_float32
        assert rotary_dim == head_size
        if inv_freq is None:
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, rotary_dim, 2, dtype=torch.float)
                    / rotary_dim
                )
            )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Derive shapes from the tensors being viewed to keep SymInts consistent for torch.compile.
        # This avoids FakeTensor failing to prove equality between independent symbolic dims
        # coming from positions.size(0) vs query.size(0).
        q_tokens = query.size(0)
        k_tokens = key.size(0)

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Reshape using only sizes from the target tensor for Dynamo friendliness
        query_shape = query.shape
        nheads_q = query_shape[-1] // self.head_size
        query = query.view(q_tokens, nheads_q, self.head_size)
        query = apply_rotary_emb(
            query,
            cos,
            sin,
            compute_in_float32=self.compute_in_float32,
        ).view(query_shape)

        key_shape = key.shape
        nheads_k = key_shape[-1] // self.head_size
        key = key.view(k_tokens, nheads_k, self.head_size)
        key = apply_rotary_emb(
            key,
            cos,
            sin,
            compute_in_float32=self.compute_in_float32,
        ).view(key_shape)
        return query, key


def compute_yarn_parameters(
    rotary_dim: int,
    base: float,
    original_max_position_embeddings: int,
    factor: float,
    *,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Return YaRN inverse frequencies and attention scaling.

    The arithmetic follows Transformers 4.49 ``_compute_yarn_parameters``.
    Keeping it local avoids coupling the runtime kernel to a particular
    Transformers model class while still making the implementation testable
    against the reference.
    """

    if rotary_dim <= 2 or rotary_dim % 2:
        raise ValueError("YaRN rotary_dim must be an even integer greater than 2")
    if original_max_position_embeddings <= 0:
        raise ValueError("original_max_position_embeddings must be positive")
    if factor <= 1.0:
        raise ValueError("YaRN factor must be greater than 1")
    if beta_fast <= 0 or beta_slow <= 0 or beta_fast <= beta_slow:
        raise ValueError("YaRN requires beta_fast > beta_slow > 0")

    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    def correction_dim(num_rotations: float) -> float:
        return (
            rotary_dim
            * math.log(
                original_max_position_embeddings
                / (num_rotations * 2.0 * math.pi)
            )
            / (2.0 * math.log(base))
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    ramp = torch.arange(rotary_dim // 2, dtype=torch.float32)
    ramp = ((ramp - low) / (high - low)).clamp(0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_freq_interpolation * (1.0 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, float(attention_factor)


@lru_cache(16)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_type: str,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    attention_factor: float | None,
    compute_in_float32: bool,
) -> RotaryEmbedding:
    inv_freq = None
    attention_scaling = 1.0
    if rope_type == "yarn":
        inv_freq, attention_scaling = compute_yarn_parameters(
            rotary_dim,
            base,
            original_max_position_embeddings,
            factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            attention_factor=attention_factor,
        )
    elif rope_type != "default":
        raise ValueError(f"unsupported RoPE scaling type: {rope_type}")
    return RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        compute_in_float32=compute_in_float32,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    compute_in_float32: bool = True,
) -> RotaryEmbedding:
    scaling = dict(rope_scaling or {})
    rope_type = str(
        scaling.pop("rope_type", scaling.pop("type", "default"))
    ).lower()
    if scaling and rope_type == "default":
        raise ValueError("RoPE scaling parameters require a scaling type")
    factor = float(scaling.pop("factor", 1.0))
    original_max = int(
        scaling.pop("original_max_position_embeddings", max_position)
    )
    beta_fast = float(scaling.pop("beta_fast", 32.0))
    beta_slow = float(scaling.pop("beta_slow", 1.0))
    raw_attention_factor = scaling.pop("attention_factor", None)
    attention_factor = (
        None
        if raw_attention_factor is None
        else float(raw_attention_factor)
    )
    if scaling:
        names = ", ".join(sorted(scaling))
        raise ValueError(f"unsupported RoPE scaling parameters: {names}")
    return _get_rope_cached(
        head_size,
        rotary_dim,
        max_position,
        base,
        rope_type,
        factor,
        original_max,
        beta_fast,
        beta_slow,
        attention_factor,
        compute_in_float32,
    )
