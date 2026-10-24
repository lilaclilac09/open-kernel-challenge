# okc: problem=muse_bf16_proj
"""BF16 projections v5：冲向 ~1.55× 的第一注（Triton split-K + 融合 router）。

相对 v4（check 1.057× / 4568 µs；scored 板上 1.055×）：
布局/pretranspose 已到平台期。领导者 ~1.55× ≈ 3119 µs，差的是小 M
带宽墙下的 CTA 占用（cuBLAS splitK + reduce 两 kernel，且小 grid）。

本版下注：
1. 非 router、m <= 128：Triton split-K GEMM（FP32 atomic 累加再 cast BF16），
   prepare() 里按 (M,N,K) autotune + warmup，保证 CUDA graph 里不再调参。
2. router m > 7：单 kernel 融合 hi+lo（x 只读一次，FP32 累加），替代 2×mm + add。
3. router m <= 7、以及 m >= 192 / Triton 失败回退：保持 v4（dispatch / mm / FP32）。

有意不做：顺序 chunk torch.mm「假 split-K」；naive GEMV（0.29× 已证伪）。
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor  # bfloat16 [8192, 1024]
    pooler: torch.Tensor  # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor  # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 7
# Triton only where cuBLAS grid is starved; large-m keeps dispatch (v3 lesson).
TRITON_MAX_M = 128


def _configs_gemm():
    cfgs = []
    for bm, bn, bk, sk, warps, stages in [
        # small M, fat K — fill H200 SMs via SPLIT_K
        (16, 64, 64, 8, 4, 3),
        (16, 128, 64, 8, 4, 4),
        (16, 64, 128, 8, 8, 3),
        (16, 128, 128, 4, 8, 4),
        (32, 64, 64, 4, 4, 3),
        (32, 128, 64, 4, 4, 4),
        (32, 128, 128, 2, 8, 3),
        (64, 64, 64, 2, 4, 3),
        (64, 128, 64, 2, 4, 4),
        # m=1-ish: BLOCK_M=16 still covers via masking; extra SK
        (16, 256, 64, 8, 8, 3),
        (16, 64, 64, 16, 4, 3),
        (16, 128, 64, 16, 4, 4),
        (16, 64, 128, 16, 8, 3),
        # no split-K baselines
        (16, 128, 64, 1, 4, 3),
        (32, 128, 64, 1, 4, 4),
        (64, 128, 64, 1, 4, 4),
    ]:
        cfgs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


def _configs_router():
    cfgs = []
    for bm, bn, bk, sk, warps, stages in [
        (16, 64, 64, 4, 4, 3),
        (16, 64, 128, 4, 4, 4),
        (16, 128, 64, 4, 4, 3),
        (16, 128, 128, 2, 8, 3),
        (32, 64, 64, 2, 4, 3),
        (32, 128, 64, 2, 4, 4),
        (16, 64, 64, 8, 4, 3),
        (16, 128, 64, 8, 4, 4),
        (16, 256, 64, 4, 8, 3),
        (32, 128, 128, 1, 4, 3),
    ]:
        cfgs.append(
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


@triton.autotune(configs=_configs_gemm(), key=["M", "N", "K"])
@triton.jit
def _gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = num_pid_m * num_pid_n

    pid_sk = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for _ in range(k_tiles):
        k_mask = rk < K
        a = tl.load(
            a_ptrs,
            mask=(rm[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        rk += BLOCK_K * SPLIT_K
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


@triton.autotune(configs=_configs_router(), key=["M", "N", "K"])
@triton.jit
def _router_hilo_kernel(
    a_ptr,
    hi_ptr,
    lo_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_hk,
    stride_hn,
    stride_lk,
    stride_ln,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_mn = num_pid_m * num_pid_n

    pid_sk = pid // num_pid_mn
    pid_mn = pid % num_pid_mn
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    hi_ptrs = hi_ptr + rk[:, None] * stride_hk + rn[None, :] * stride_hn
    lo_ptrs = lo_ptr + rk[:, None] * stride_lk + rn[None, :] * stride_ln

    k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for _ in range(k_tiles):
        k_mask = rk < K
        a = tl.load(
            a_ptrs,
            mask=(rm[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        hi = tl.load(
            hi_ptrs,
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=0.0,
        )
        lo = tl.load(
            lo_ptrs,
            mask=k_mask[:, None] & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, hi)
        acc += tl.dot(a, lo)
        rk += BLOCK_K * SPLIT_K
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        hi_ptrs += BLOCK_K * SPLIT_K * stride_hk
        lo_ptrs += BLOCK_K * SPLIT_K * stride_lk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


def triton_gemm_bf16(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    """x [M,K] bf16 @ wt [K,N] bf16 -> [M,N] bf16. Split-K accumulates in FP32."""
    M, K = x.shape
    N = wt.shape[1]
    # FP32 buffer so split-K atomics stay accurate enough for 1% L2.
    out_f = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"])
            * triton.cdiv(N, meta["BLOCK_N"])
            * meta["SPLIT_K"],
        )

    _gemm_splitk_kernel[grid](
        x,
        wt,
        out_f,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        wt.stride(0),
        wt.stride(1),
        out_f.stride(0),
        out_f.stride(1),
    )
    return out_f.to(torch.bfloat16)


def triton_router_hilo(
    x: torch.Tensor, hi_t: torch.Tensor, lo_t: torch.Tensor
) -> torch.Tensor:
    """Fused out = x@hi_t + x@lo_t in FP32."""
    M, K = x.shape
    N = hi_t.shape[1]
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"])
            * triton.cdiv(N, meta["BLOCK_N"])
            * meta["SPLIT_K"],
        )

    _router_hilo_kernel[grid](
        x,
        hi_t,
        lo_t,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        hi_t.stride(0),
        hi_t.stride(1),
        lo_t.stride(0),
        lo_t.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[f"{name}_t"] = state[name].t().contiguous()
    ut = state["unpooler_t"]
    state["unpooler_t_1024"] = ut[:, :1024].contiguous()
    state["unpooler_t_8192"] = ut[:, :8192].contiguous()
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    # Warm / autotune every scored + correctness shape so CUDA-graph capture
    # never hits the autotuner.
    device = state["qkv_gate"].device
    warm_ms = [1, 5, 7, 8, 9, 16, 32, 33, 128, 130, 256, 512, 1000]
    layer_nk = {
        "qkv_gate": (2304, 8192),
        "o": (8192, 1024),
        "pooler": (4096, 8192),
        "unpooler": (8192, 4096),
    }
    for layer, (n_full, k) in layer_nk.items():
        wt_full = state[f"{layer}_t"]
        for m in warm_ms:
            if m > TRITON_MAX_M:
                continue
            n = 1024 if (layer == "unpooler" and m <= 8) else n_full
            if layer == "unpooler" and n == 1024:
                wt = state["unpooler_t_1024"]
            elif layer == "unpooler" and n == 8192:
                wt = state["unpooler_t_8192"]
            else:
                wt = wt_full if n == wt_full.shape[1] else wt_full[:, :n]
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            try:
                triton_gemm_bf16(x, wt)
            except Exception:
                pass
    for m in warm_ms:
        if m <= ROUTER_FP32_MAX_TOKENS:
            continue
        x = torch.zeros((m, 8192), device=device, dtype=torch.bfloat16)
        try:
            triton_router_hilo(x, state["router_hi_t"], state["router_lo_t"])
        except Exception:
            pass
    torch.cuda.synchronize()
    return state


def _w_t(state: dict[str, Any], layer: str, n: int) -> torch.Tensor:
    if layer == "unpooler":
        if n == 1024:
            return state["unpooler_t_1024"]
        if n == 8192:
            return state["unpooler_t_8192"]
        return state["unpooler_t"][:, :n]
    wt = state[f"{layer}_t"]
    if n == wt.shape[1]:
        return wt
    return wt[:, :n]


def _v4_bf16(x: torch.Tensor, state: dict[str, Any], layer: str, n: int, m: int):
    if m >= 192:
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    wt = _w_t(state, layer, n)
    if wt.is_contiguous() and wt.shape[1] == n:
        return torch.mm(x, wt)
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"].contiguous()
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]

    if layer != "router":
        if m <= TRITON_MAX_M:
            wt = _w_t(state, layer, n)
            if wt.is_contiguous() and wt.shape[1] == n:
                return triton_gemm_bf16(x, wt)
        return _v4_bf16(x, state, layer, n, m)

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    return triton_router_hilo(x, state["router_hi_t"], state["router_lo_t"])
