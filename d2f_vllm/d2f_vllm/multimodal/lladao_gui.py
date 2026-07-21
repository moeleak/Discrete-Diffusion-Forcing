from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_model
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional
from transformers import AutoTokenizer


class _VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.size(0)
        shape = (1, length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return self.out_proj(
            output.transpose(1, 2).reshape(length, self.hidden_size)
        )


class _VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class _VisionEncoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        eps = float(config.get("layer_norm_eps", 1e-6))
        self.self_attn = _VisionAttention(
            hidden_size, int(config["num_attention_heads"])
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _VisionMLP(hidden_size, int(config["intermediate_size"]))
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states)
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class _VisionEmbeddings(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        patch_size = int(config["patch_size"])
        channels = int(config.get("num_channels", 3))
        self.patch_embedding = nn.Linear(channels * patch_size**2, hidden_size)
        positions = (int(config["image_size"]) // patch_size) ** 2
        self.position_embedding = nn.Embedding(positions, hidden_size)

    def forward(
        self, patches: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.patch_embedding(patches) + self.position_embedding(position_ids)


class _VisionEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _VisionEncoderLayer(config)
            for _ in range(int(config["num_hidden_layers"]))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _VisionTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = _VisionEmbeddings(config)
        self.encoder = _VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            int(config["hidden_size"]),
            eps=float(config.get("layer_norm_eps", 1e-6)),
        )

    def forward(
        self, patches: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embeddings(patches, position_ids)
        return self.post_layernorm(self.encoder(hidden_states))


class _VisionModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.vision_model = _VisionTransformer(config)

    def forward(
        self, patches: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.vision_model(patches, position_ids)


class _Connector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.fc1(hidden_states), approximate="tanh")
        return self.fc2(hidden_states)


class _PositionEmbedding(nn.Module):
    def __init__(self, positions: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.empty(positions, hidden_size), requires_grad=False
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class _VisionPrefixModules(nn.Module):
    def __init__(self, vision_config: dict, language_hidden_size: int) -> None:
        super().__init__()
        self.vit_model = _VisionModel(vision_config)
        self.connector = _Connector(
            int(vision_config["hidden_size"]), language_hidden_size
        )
        self.vit_pos_embed = _PositionEmbedding(70**2, language_hidden_size)


@dataclass
class LLaDAOGuiPrefix:
    image_ids: list[int]
    image_positions: list[int]
    image_embeddings: torch.Tensor
    prompt_ids: list[int]
    prompt_positions: list[int]

    @property
    def length(self) -> int:
        return len(self.image_ids) + len(self.prompt_ids)


class LLaDAOGuiPrefixEncoder:
    """Exact LLaDA-o GUI preprocessing with a native SDPA SigLIP encoder."""

    def __init__(
        self,
        model_path: str | Path,
        token_embedding: nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.token_embedding = token_embedding
        vision_config = json.loads(
            (self.model_path / "vision_config.json").read_text()
        )
        language_config = json.loads((self.model_path / "config.json").read_text())
        previous_dtype = torch.get_default_dtype()
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("cpu")
            torch.set_default_dtype(dtype)
            modules = _VisionPrefixModules(
                vision_config, int(language_config["hidden_size"])
            )
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)
        missing, unexpected = load_model(
            modules,
            str(self.model_path / "vision.safetensors"),
            strict=True,
            device="cpu",
        )
        if missing or unexpected:
            raise RuntimeError(
                f"vision checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.modules = modules.to(device=device, dtype=dtype).eval()
        self.patch_size = int(vision_config["patch_size"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), use_fast=True, trust_remote_code=False
        )
        self.bos_token_id = self._token_id("<|startoftext|>")
        self.eos_token_id = self._token_id("<|endoftext|>")
        self.start_image_id = self._token_id("<|vision_start|>")
        self.end_image_id = self._token_id("<|vision_end|>")

    def _token_id(self, token: str) -> int:
        token_id = int(self.tokenizer.convert_tokens_to_ids(token))
        if token_id < 0 or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"runtime tokenizer does not define {token}")
        return token_id

    @staticmethod
    def _make_divisible(value: float, stride: int) -> int:
        return max(stride, int(round(value / stride) * stride))

    def _resize(self, image: Image.Image) -> Image.Image:
        max_size, min_size, stride, max_pixels = 980, 378, 14, 2_007_040
        width, height = image.size
        scale = min(max_size / max(width, height), 1.0)
        scale = max(scale, min_size / min(width, height))
        new_width = self._make_divisible(round(width * scale), stride)
        new_height = self._make_divisible(round(height * scale), stride)
        if new_width * new_height > max_pixels:
            scale = max_pixels / (new_width * new_height)
            new_width = self._make_divisible(round(new_width * scale), stride)
            new_height = self._make_divisible(round(new_height * scale), stride)
        if max(new_width, new_height) > max_size:
            scale = max_size / max(new_width, new_height)
            new_width = self._make_divisible(round(new_width * scale), stride)
            new_height = self._make_divisible(round(new_height * scale), stride)
        return tv_functional.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _patchify(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._resize(image.convert("RGB"))
        tensor = tv_functional.pil_to_tensor(image).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        channels, height, width = tensor.shape
        patch = self.patch_size
        tensor = tensor.reshape(
            channels, height // patch, patch, width // patch, patch
        )
        patches = torch.einsum("chpwq->hwpqc", tensor).reshape(
            -1, patch**2 * channels
        )
        rows = torch.arange(height // patch)
        columns = torch.arange(width // patch)
        positions = (rows[:, None] * 70 + columns).flatten()
        return patches, positions

    @torch.inference_mode()
    def encode(self, image: Image.Image, prompt: str) -> LLaDAOGuiPrefix:
        patches, vision_positions = self._patchify(image)
        patches = patches.to(device=self.device, dtype=self.dtype)
        vision_positions = vision_positions.to(device=self.device)
        vision = self.modules.vit_model(patches, vision_positions)
        vision = self.modules.connector(vision)
        vision = vision + self.modules.vit_pos_embed(vision_positions)

        boundary_ids = torch.tensor(
            [self.start_image_id, self.end_image_id],
            dtype=torch.long,
            device=self.device,
        )
        boundary = self.token_embedding(boundary_ids)
        image_embeddings = torch.cat(
            (boundary[:1], vision.to(boundary.dtype), boundary[1:]), dim=0
        )
        image_ids = [self.start_image_id] + [0] * vision.size(0) + [self.end_image_id]
        prompt_ids = [
            self.bos_token_id,
            *self.tokenizer.encode(prompt, add_special_tokens=False),
            self.eos_token_id,
        ]
        return LLaDAOGuiPrefix(
            image_ids=image_ids,
            image_positions=[0] * len(image_ids),
            image_embeddings=image_embeddings,
            prompt_ids=prompt_ids,
            prompt_positions=list(range(1, 1 + len(prompt_ids))),
        )
