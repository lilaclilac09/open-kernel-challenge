# okc: problem=muse_bf16_proj
"""BF16 projections v6c：v6b + unpooler m=1 也走 GEMV。

相对 v6b（板上 #685 · 1.071×）：只改一处——m=1 时 unpooler 不再排除，
和 qkv/o/pooler 一样走 warp-per-output GEMV（N=1024 → 128 block）。
其余路径与 v6b/v4 一字相同（torch.mm / m>=192 dispatch / router 阈值 7）。

load_inline 必须带非空 cpp_sources 前向声明，否则 pybind 静默绑不上
gemv_bf16（v5/v7 教训）。编译失败则 gemv_mod=None，自动退回 v4。
"""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from torch.utils.cpp_extension import load_inline as _load_inline
except Exception:  # pragma: no cover - torch without cpp_extension
    _load_inline = None

# load_inline pybind main.cpp only sees cpp_sources — empty string => undeclared
# (v5/v7 silent nvjet-only fallback). Forward-decl so gemv_bf16 binds.
_CPP_SRC = r"""
#include <torch/extension.h>
void gemv_bf16(torch::Tensor X, torch::Tensor W, torch::Tensor Out, int64_t N, int64_t K);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// GEMV for m == 1 only. One warp computes one output element:
//   out[n] = sum_k x[k] * W[n][k],  W is [N, K] row-major bf16.
// Block = 256 threads = 8 warps -> 8 outputs per block.
// Grid  = ceil(N / 8) blocks (qkv_gate 288, o 1024, pooler 512, unpooler 128).
// Each lane loads one uint4 (8 bf16) per step, so one warp step covers
// 32 * 8 = 256 consecutive k elements with contiguous 512-byte reads.
// No shared memory, no split-K, no second reduce kernel: the warp shuffle
// at the end IS the reduction. FP32 accumulation, bf16 output.
__global__ void __launch_bounds__(256) gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Out,
    int N, int K)
{
    const int warp = threadIdx.x >> 5;       // 0..7 inside the block
    const int lane = threadIdx.x & 31;       // 0..31 inside the warp
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;

    const __nv_bfloat16* wrow = W + (size_t)n * K;
    float acc = 0.0f;
    // K is a multiple of 256 (checked on the Python side), so this loop
    // covers every k exactly once, with no tail handling needed.
    for (int k0 = 0; k0 < K; k0 += 256) {
        const int k = k0 + lane * 8;
        const uint4 xv = *reinterpret_cast<const uint4*>(X + k);
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k);
        // Each uint32 holds one bf162 pair; the pair sum is order-free
        // (a0*b0 + a1*b1), so low/high interpretation cannot change it.
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t xa = (p == 0) ? xv.x : (p == 1) ? xv.y : (p == 2) ? xv.z : xv.w;
            const uint32_t wa = (p == 0) ? wv.x : (p == 1) ? wv.y : (p == 2) ? wv.z : wv.w;
            const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa));
            const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wa));
            acc += af.x * wf.x + af.y * wf.y;
        }
    }
    // Warp reduction: sum the 32 lane partials into lane 0.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        Out[n] = __float2bfloat16(acc);
    }
}

void gemv_bf16(torch::Tensor X, torch::Tensor W, torch::Tensor Out, int64_t N, int64_t K) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && Out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16, "X/W must be bf16");
    TORCH_CHECK(Out.scalar_type() == at::kBFloat16, "Out must be bf16");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && Out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(X.dim() == 2 && X.size(0) == 1 && X.size(1) == K, "X must be [1, K]");
    TORCH_CHECK(W.numel() >= N * K, "W too small for N x K");
    TORCH_CHECK(Out.numel() >= N, "Out too small for N");
    dim3 grid((unsigned)((N + 7) / 8));
    dim3 block(256);
    auto stream = at::cuda::getCurrentCUDAStream();
    gemv_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr()),
        (int)N, (int)K);
}
"""


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


def _get_gemv_mod() -> Any:
    if _load_inline is None:
        return None
    try:
        return _load_inline(
            name="bf16_gemv_v6c",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gemv_bf16"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
    except Exception:
        return None


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
    # GEMV 编译只在 prepare（不计时）做一次；失败则 gemv_mod 为 None，
    # kernel_fn 自动退回 v4 路，不会炸掉整个提交。
    state["gemv_mod"] = _get_gemv_mod()
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


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"].contiguous()
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]
    k = cfg["k"]

    if layer != "router":
        # m=1：全部非 router 层走 GEMV（含 unpooler）
        mod = state.get("gemv_mod")
        if mod is not None and m == 1 and k % 256 == 0:
            try:
                # 传原始 W（[N, K] 行主序），不要传预转置版；前 n 行切片。
                b = state[layer][:n].contiguous()
                out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
                mod.gemv_bf16(x, b, out, n, k)
                return out
            except Exception:
                pass  # 同步检查失败时往下走 v4 路

        # ---- 以下与 v4 一字相同 ----
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
