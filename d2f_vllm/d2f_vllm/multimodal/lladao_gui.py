from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_model
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional
from transformers import AutoTokenizer
from d2f_vllm.utils.vllm_flash import flash_attn_varlen_func


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
        shape = (length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape)
        if flash_attn_varlen_func is not None and hidden_states.is_cuda:
            cu_seqlens = torch.tensor(
                [0, length], dtype=torch.int32, device=hidden_states.device
            )
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                max_seqlen_q=length,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_k=length,
                cu_seqlens_k=cu_seqlens,
                causal=False,
            )
        else:
            output = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(0).transpose(0, 1)
        return self.out_proj(output.reshape(length, self.hidden_size))


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


@dataclass(frozen=True)
class LLaDAOGuiImageSpan:
    token_start: int
    patch_start: int
    patch_end: int
    token_end: int
    grid_height: int
    grid_width: int
    source_box: tuple[int, int, int, int]

    @property
    def patch_count(self) -> int:
        return self.patch_end - self.patch_start


@dataclass
class LLaDAOGuiPrefix:
    image_ids: list[int]
    image_positions: list[int]
    image_embeddings: torch.Tensor
    image_spans: list[LLaDAOGuiImageSpan]
    source_width: int
    source_height: int
    prompt_ids: list[int]
    prompt_positions: list[int]
    position_mode: str = "native"

    @property
    def length(self) -> int:
        return len(self.image_ids) + len(self.prompt_ids)

    @property
    def image_patch_count(self) -> int:
        return sum(span.patch_count for span in self.image_spans)


def full_page_tile_boxes(
    width: int,
    height: int,
    tile_size: int = 980,
) -> list[tuple[int, int, int, int]]:
    """Split a screenshot into deterministic non-overlapping row-major tiles."""

    if width <= 0 or height <= 0:
        raise ValueError("full-page image dimensions must be positive")
    if tile_size <= 0 or tile_size > 980:
        raise ValueError("full-page tile_size must be in [1, 980]")
    return [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    ]


def build_multimodal_position_ids(
    image_token_lengths: Sequence[int],
    prompt_token_length: int,
    *,
    mode: str = "native",
) -> tuple[list[int], list[int]]:
    """Build native or token-sequential LLM RoPE positions.

    LLaDA-o natively gives every token from one image a shared global
    position.  The opt-in ``sequential`` mode instead assigns every visual
    boundary/patch token its own absolute position and starts the text prompt
    after the complete visual prefix.  This is intended for controlled
    long-RoPE experiments, not as a silent change to native inference.
    """

    if mode not in {"native", "sequential"}:
        raise ValueError(
            "multimodal position mode must be one of: native, sequential"
        )
    lengths = [int(length) for length in image_token_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("image_token_lengths must contain positive values")
    prompt_length = int(prompt_token_length)
    if prompt_length <= 0:
        raise ValueError("prompt_token_length must be positive")

    image_positions: list[int] = []
    token_cursor = 0
    for image_index, length in enumerate(lengths):
        if mode == "native":
            image_positions.extend([image_index] * length)
        else:
            image_positions.extend(range(token_cursor, token_cursor + length))
        token_cursor += length
    prompt_start = len(lengths) if mode == "native" else token_cursor
    prompt_positions = list(range(prompt_start, prompt_start + prompt_length))
    return image_positions, prompt_positions


class LLaDAOGuiPrefixEncoder:
    """Exact LLaDA-o GUI preprocessing with a native SigLIP encoder."""

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
        self._ensure_special_tokens()
        self.bos_token_id = self._token_id("<|startoftext|>")
        self.eos_token_id = self._token_id("<|endoftext|>")
        self.start_image_id = self._token_id("<|vision_start|>")
        self.end_image_id = self._token_id("<|vision_end|>")

    def _ensure_special_tokens(self) -> None:
        existing = set(self.tokenizer.all_special_tokens)
        ordered = (
            "<|startoftext|>",
            "<|endoftext|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
        additions = [token for token in ordered if token not in existing]
        if additions:
            self.tokenizer.add_tokens(additions)

    def _token_id(self, token: str) -> int:
        value = self.tokenizer.convert_tokens_to_ids(token)
        if value is None:
            raise ValueError(f"runtime tokenizer does not define {token}")
        token_id = int(value)
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

    def _patchify(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        image = self._resize(image.convert("RGB"))
        return self._patchify_exact(image)

    def _patchify_exact(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Patchify without resizing, padding only the right and bottom edges."""

        tensor = tv_functional.pil_to_tensor(image).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        channels, height, width = tensor.shape
        patch = self.patch_size
        padded_height = ((height + patch - 1) // patch) * patch
        padded_width = ((width + patch - 1) // patch) * patch
        if padded_height > 980 or padded_width > 980:
            raise ValueError(
                "exact image tile exceeds the trained 70x70 ViT position grid"
            )
        if padded_height != height or padded_width != width:
            tensor = F.pad(
                tensor,
                (0, padded_width - width, 0, padded_height - height),
                value=0.0,
            )
            height, width = padded_height, padded_width
        tensor = tensor.reshape(
            channels, height // patch, patch, width // patch, patch
        )
        patches = torch.einsum("chpwq->hwpqc", tensor).reshape(
            -1, patch**2 * channels
        )
        rows = torch.arange(height // patch)
        columns = torch.arange(width // patch)
        positions = (rows[:, None] * 70 + columns).flatten()
        return patches, positions, height // patch, width // patch

    def _encode_images(
        self,
        images: list[tuple[Image.Image, tuple[int, int, int, int]]],
        prompt: str,
        *,
        resize: bool,
        source_size: tuple[int, int],
        position_mode: str = "native",
    ) -> LLaDAOGuiPrefix:
        if not images:
            raise ValueError("at least one image is required")
        boundary_ids = torch.tensor(
            [self.start_image_id, self.end_image_id],
            dtype=torch.long,
            device=self.device,
        )
        boundary = self.token_embedding(boundary_ids)
        image_ids: list[int] = []
        image_embeddings: list[torch.Tensor] = []
        image_spans: list[LLaDAOGuiImageSpan] = []
        image_token_lengths: list[int] = []
        token_cursor = 0

        for image, source_box in images:
            patchify = self._patchify if resize else self._patchify_exact
            patches, vision_positions, grid_height, grid_width = patchify(
                image.convert("RGB")
            )
            patches = patches.to(device=self.device, dtype=self.dtype)
            vision_positions = vision_positions.to(device=self.device)
            vision = self.modules.vit_model(patches, vision_positions)
            vision = self.modules.connector(vision)
            vision = vision + self.modules.vit_pos_embed(vision_positions)
            encoded = torch.cat(
                (boundary[:1], vision.to(boundary.dtype), boundary[1:]), dim=0
            )
            patch_count = int(vision.size(0))
            span = LLaDAOGuiImageSpan(
                token_start=token_cursor,
                patch_start=token_cursor + 1,
                patch_end=token_cursor + 1 + patch_count,
                token_end=token_cursor + patch_count + 2,
                grid_height=grid_height,
                grid_width=grid_width,
                source_box=source_box,
            )
            image_spans.append(span)
            image_embeddings.append(encoded)
            image_ids.extend(
                [self.start_image_id] + [0] * patch_count + [self.end_image_id]
            )
            image_token_lengths.append(patch_count + 2)
            token_cursor = span.token_end

        prompt_ids = [
            self.bos_token_id,
            *self.tokenizer.encode(prompt, add_special_tokens=False),
            self.eos_token_id,
        ]
        image_positions, prompt_positions = build_multimodal_position_ids(
            image_token_lengths,
            len(prompt_ids),
            mode=position_mode,
        )
        return LLaDAOGuiPrefix(
            image_ids=image_ids,
            image_positions=image_positions,
            image_embeddings=torch.cat(image_embeddings, dim=0),
            image_spans=image_spans,
            source_width=int(source_size[0]),
            source_height=int(source_size[1]),
            prompt_ids=prompt_ids,
            prompt_positions=prompt_positions,
            position_mode=position_mode,
        )

    @torch.inference_mode()
    def encode(self, image: Image.Image, prompt: str) -> LLaDAOGuiPrefix:
        width, height = image.size
        return self._encode_images(
            [(image, (0, 0, width, height))],
            prompt,
            resize=True,
            source_size=(width, height),
        )

    @torch.inference_mode()
    def encode_full_page(
        self,
        image: Image.Image,
        prompt: str,
        *,
        tile_size: int = 980,
        position_mode: str = "native",
    ) -> LLaDAOGuiPrefix:
        image = image.convert("RGB")
        width, height = image.size
        boxes = full_page_tile_boxes(width, height, tile_size)
        images = [(image.crop(box), box) for box in boxes]
        return self._encode_images(
            images,
            prompt,
            resize=False,
            source_size=(width, height),
            position_mode=position_mode,
        )
