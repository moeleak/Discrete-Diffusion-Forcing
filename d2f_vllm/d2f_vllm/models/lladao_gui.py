from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from d2f_vllm.layers.activation import SiluAndMul
from d2f_vllm.layers.attention.attention_v4 import Attention
from d2f_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from d2f_vllm.layers.layernorm import RMSNorm
from d2f_vllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from d2f_vllm.layers.rotary_embedding import get_rope
from d2f_vllm.models.config.lladao_gui.configuration_lladao_gui import (
    LLaDAOGuiConfig,
)


if os.environ.get("TRITON_INTERPRET") == "1":
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = True
    torch.backends.optimized_mode = False


class _ExactLoRAMixin:
    def _init_exact_lora(
        self,
        rank: int,
        alpha: float,
        in_features: int,
        out_features: int,
    ) -> None:
        if rank <= 0:
            raise ValueError("exact runtime LoRA rank must be positive")
        if self.tp_size != 1:
            raise ValueError("exact LLaDA-o runtime LoRA currently requires TP=1")
        self.lora_scale = float(alpha) / int(rank)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=torch.float32),
            requires_grad=False,
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch.float32),
            requires_grad=False,
        )

    def _apply_exact_lora(
        self,
        hidden_states: torch.Tensor,
        base_output: torch.Tensor,
        *,
        lora_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lora_input is None:
            lora_input = hidden_states.to(self.lora_A.dtype)
        lora_hidden = F.linear(lora_input, self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B)
        if self.lora_scale != 1.0:
            lora_output = lora_output * self.lora_scale
        return (base_output + lora_output).to(base_output.dtype)


class _ExactLoRAColumnParallelLinear(_ExactLoRAMixin, ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        rank: int,
        alpha: float,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, bias=bias)
        self._init_exact_lora(
            rank, alpha, self.input_size, self.output_size_per_partition
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        lora_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_output = F.linear(hidden_states, self.weight, self.bias)
        return self._apply_exact_lora(
            hidden_states, base_output, lora_input=lora_input
        )


class _ExactLoRARowParallelLinear(_ExactLoRAMixin, RowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        rank: int,
        alpha: float,
        bias: bool = False,
    ) -> None:
        super().__init__(input_size, output_size, bias=bias)
        self._init_exact_lora(
            rank, alpha, self.input_size_per_partition, self.output_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        lora_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_output = F.linear(hidden_states, self.weight, self.bias)
        return self._apply_exact_lora(
            hidden_states, base_output, lora_input=lora_input
        )


class LLaDAOGuiAttention(nn.Module):
    def __init__(self, config: LLaDAOGuiConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("attention heads must be divisible by tensor parallel size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.scaling = self.head_dim**-0.5
        bias = bool(config.attention_bias)
        lora_rank = int(config.runtime_lora_rank)
        self.exact_lora = lora_rank > 0
        if lora_rank:
            def projection(output_size: int) -> _ExactLoRAColumnParallelLinear:
                return _ExactLoRAColumnParallelLinear(
                    config.hidden_size,
                    output_size,
                    rank=lora_rank,
                    alpha=float(config.runtime_lora_alpha),
                    bias=bias,
                )

            self.q_proj = projection(self.total_num_heads * self.head_dim)
            self.k_proj = projection(self.total_num_kv_heads * self.head_dim)
            self.v_proj = projection(self.total_num_kv_heads * self.head_dim)
            self.o_proj = _ExactLoRARowParallelLinear(
                self.total_num_heads * self.head_dim,
                config.hidden_size,
                rank=lora_rank,
                alpha=float(config.runtime_lora_alpha),
                bias=False,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size, self.total_num_heads * self.head_dim, bias=bias
            )
            self.k_proj = ColumnParallelLinear(
                config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=bias
            )
            self.v_proj = ColumnParallelLinear(
                config.hidden_size, self.total_num_kv_heads * self.head_dim, bias=bias
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim, config.hidden_size, bias=False
            )
        if not config.qk_norm:
            raise ValueError("the GUI-grounding LLaDA-o checkpoint requires qk_norm=True")
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            compute_in_float32=False,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            "diffusion_lm",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.exact_lora:
            lora_input = hidden_states.to(self.q_proj.lora_A.dtype)
            query = self.q_proj(
                hidden_states, lora_input=lora_input
            ).view(-1, self.num_heads, self.head_dim)
            key = self.k_proj(
                hidden_states, lora_input=lora_input
            ).view(-1, self.num_kv_heads, self.head_dim)
            value = self.v_proj(hidden_states, lora_input=lora_input)
        else:
            query = self.q_proj(hidden_states).view(
                -1, self.num_heads, self.head_dim
            )
            key = self.k_proj(hidden_states).view(
                -1, self.num_kv_heads, self.head_dim
            )
            value = self.v_proj(hidden_states)
        query = self.q_norm(query).reshape(-1, self.num_heads * self.head_dim)
        key = self.k_norm(key).reshape(-1, self.num_kv_heads * self.head_dim)
        query, key = self.rotary_emb(positions, query, key)
        return self.o_proj(self.attn(query, key, value, mask))


class LLaDAOGuiMLP(nn.Module):
    def __init__(self, config: LLaDAOGuiConfig) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported LLaDA-o activation: {config.hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(self.act_fn(torch.cat((gate, up), dim=-1)))


class LLaDAOGuiDecoderLayer(nn.Module):
    def __init__(self, config: LLaDAOGuiConfig) -> None:
        super().__init__()
        self.self_attn = LLaDAOGuiAttention(config)
        self.mlp = LLaDAOGuiMLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            residual_in_fp32=False,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            residual_in_fp32=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, mask)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LLaDAOGuiModel(nn.Module):
    def __init__(self, config: LLaDAOGuiConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LLaDAOGuiDecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            residual_in_fp32=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (input_embeds is None):
            raise ValueError("provide exactly one of input_ids or input_embeds")
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )
        if hidden_states.size(0) != positions.numel():
            raise ValueError("embedding and position lengths do not match")
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, mask
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LLaDAOGuiForDiffusionLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config: LLaDAOGuiConfig) -> None:
        super().__init__()
        self.model = LLaDAOGuiModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, model_type="diffusion_lm"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, mask, input_embeds=input_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
