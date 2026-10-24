# okc: problem=muse_bf16_proj
"""BF16 projections: Triton GEMV for M==1, cuBLAS/_bf16 dispatch otherwise; router hi/lo."""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


@triton.jit
def _gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y[n] = sum_k x[k] * W[n, k]; W is [N, K]."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for kk in range(0, num_k):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)


def _gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [1, K] BF16, w: [N, K] BF16 -> [1, N] BF16."""
    assert x.shape[0] == 1
    k = x.shape[1]
    n = w.shape[0]
    out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
    block_n = 64
    block_k = 256
    grid = (triton.cdiv(n, block_n),)
    _gemv_kernel[grid](
        x.view(-1),
        w,
        out.view(-1),
        n,
        k,
        w.stride(0),
        w.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


@triton.jit
def _gemm_small_m_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for kk in range(0, num_k):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _gemm_small_m(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, K], w: [N, K] -> [M, N], for small M."""
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    block_m = 16 if m <= 16 else 32
    block_n = 64
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _gemm_small_m_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[name].contiguous()
        state[name] = w
        state[f"{name}_t"] = w.t().contiguous()
    # Contiguous unpooler slices for n=1024 (TP slice) and full n=8192
    u = state["unpooler"]
    state["unpooler_1024"] = u[:1024].contiguous()
    state["unpooler_8192"] = u[:8192].contiguous()
    # Full contiguous packs for other layers (n equals full rows)
    state["qkv_gate_full"] = state["qkv_gate"]  # [2304, 8192]
    state["o_full"] = state["o"]                # [8192, 1024]
    state["pooler_full"] = state["pooler"]      # [4096, 8192]

    router = weights["router"]
    state["router"] = router.contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state


def _weight_nk(state: dict[str, Any], layer: str, n: int) -> torch.Tensor:
    if layer == "unpooler":
        if n == 1024:
            return state["unpooler_1024"]
        if n == 8192:
            return state["unpooler_8192"]
        return state["unpooler"][:n]
    return state[layer][:n]


def _bf16_proj(x: torch.Tensor, w: torch.Tensor, m: int) -> torch.Tensor:
    # Branch only on config m (CUDA-graph safe).
    if m == 1:
        return _gemv(x, w)
    if m <= 32:
        # Small-M Triton GEMM; fall back path still available via dispatch if needed.
        return _gemm_small_m(x, w)
    return _bf16_gemm_dispatch_impl(x, w, None)


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]

    if layer != "router":
        w = _weight_nk(state, layer, n)
        return _bf16_proj(x, w, m)

    # Always hi/lo BF16 with FP32 accumulation (legal at any m; graph-friendly).
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
