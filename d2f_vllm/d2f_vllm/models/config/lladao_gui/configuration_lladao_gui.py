from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class LLaDAOGuiConfig(PretrainedConfig):
    """Configuration for the LLaDA-o visual-understanding language path."""

    model_type = "lladao_gui"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 126464,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 16384,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        rope_scaling: dict | None = None,
        attention_bias: bool = False,
        qk_norm: bool = True,
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
