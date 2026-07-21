import os

import torch
from torch import nn


_VLLM_OPS = None


def _get_vllm_ops():
    global _VLLM_OPS
    if _VLLM_OPS is None:
        from vllm import _custom_ops

        _VLLM_OPS = _custom_ops
    return _VLLM_OPS


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        residual_in_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.backend = os.environ.get(
            "D2F_VLLM_RMS_NORM_BACKEND", "torch"
        ).lower()
        if self.backend not in {"torch", "vllm"}:
            raise ValueError(
                "D2F_VLLM_RMS_NORM_BACKEND must be 'torch' or 'vllm'"
            )
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def vllm_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ops = _get_vllm_ops()
        if residual is None:
            output = torch.empty_like(x)
            ops.rms_norm(output, x, self.weight.data, self.eps)
            return output
        ops.fused_add_rms_norm(x, residual, self.weight.data, self.eps)
        return x, residual

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if self.residual_in_fp32:
            x = x.to(torch.float32).add_(residual.to(torch.float32))
            residual = x.to(orig_dtype)
        else:
            residual = x.add(residual)
            x = residual.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "vllm" and not self.residual_in_fp32:
            return self.vllm_forward(x, residual)
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
