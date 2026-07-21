from functools import lru_cache
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
    else:
        working = x
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
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
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.compute_in_float32 = compute_in_float32
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
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


@lru_cache(8)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
    compute_in_float32: bool = True,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        compute_in_float32=compute_in_float32,
    )
    return rotary_emb
