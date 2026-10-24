# okc: problem=muse_bf16_proj
"""BF16 projections v6d：v6b 分档 + 加强版 m=1 GEMV（不碰 bmm）。

相对 v6b（板上 #687 · 1.071×）：
- Python 分档一字不动：m=1 GEMV（unpooler 除外）、m>=192 dispatch、
  其余 torch.mm、router 阈值 7。不重开 unpooler GEMV（v6c 总分回落）。
- CUDA GEMV 三处微优化（明日 check 再验）：
  1. 每 warp 算 2 个相邻输出，共用同一份 X 的 uint4 读 —— 经典 GEMV
     带宽招：X 流量减半，W 仍各读一遍。block=256 / 8 warps → 16 输出/
     block（v6b 是 8），grid = ceil(N/16)。
  2. K 步进 512（两段独立 256），双累加器叠 ILP；题面 K 都是 1024/
     4096/8192，整除无尾巴；仍保留 256 对齐的尾循环兜底。
  3. X 用 __ldg（只读缓存，跨行复用），W 用 __ldcs（streaming，
     不污染 L2）。

不重开：torch.bmm / _split_s（v6 拖垮主因）。load_inline 必须带非空
cpp_sources 前向声明（v5/v7 教训）。编译失败则 gemv_mod=None，退回 v4。

# ENABLE_UNPOOLER_GEMV = False  # v6c 试过，整盘略慢；默认关
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

// GEMV for m == 1 only. Dual-output-per-warp (v6d):
//   out[n0], out[n1] = sum_k x[k] * W[n*][k],  W is [N, K] row-major bf16.
// Block = 256 threads = 8 warps -> 16 outputs per block (2 per warp).
// Grid  = ceil(N / 16) (qkv_gate 144, o 512, pooler 256).
// Each lane loads one uint4 (8 bf16) per 256-chunk; X is shared across the
// two W rows so X traffic is halved vs v6b. Outer step is 512 (two independent
// 256-chunks) for ILP. X via __ldg, W via __ldcs. Warp shuffle reduce; FP32
// accum; bf16 store. No shared mem, no split-K, no second reduce kernel.
__device__ __forceinline__ void mad_u4(
    float& acc, const uint4 xv, const uint4 wv)
{
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        const uint32_t xa = (p == 0) ? xv.x : (p == 1) ? xv.y : (p == 2) ? xv.z : xv.w;
        const uint32_t wa = (p == 0) ? wv.x : (p == 1) ? wv.y : (p == 2) ? wv.z : wv.w;
        const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa));
        const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wa));
        acc += af.x * wf.x + af.y * wf.y;
    }
}

__global__ void __launch_bounds__(256) gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Out,
    int N, int K)
{
    const int warp = threadIdx.x >> 5;       // 0..7 inside the block
    const int lane = threadIdx.x & 31;       // 0..31 inside the warp
    const int n0 = blockIdx.x * 16 + warp * 2;
    if (n0 >= N) return;
    const int n1 = n0 + 1;
    const bool dual = (n1 < N);

    const __nv_bfloat16* wrow0 = W + (size_t)n0 * K;
    const __nv_bfloat16* wrow1 = dual ? (W + (size_t)n1 * K) : wrow0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // K is a multiple of 256 (checked on the Python side). Prefer 512-step
    // with two independent chunks for ILP; fall back to a single 256-tail.
    int k0 = 0;
    for (; k0 + 512 <= K; k0 += 512) {
        const int ka = k0 + lane * 8;
        const int kb = k0 + 256 + lane * 8;
        const uint4 xa = __ldg(reinterpret_cast<const uint4*>(X + ka));
        const uint4 xb = __ldg(reinterpret_cast<const uint4*>(X + kb));
        const uint4 wa0 = __ldcs(reinterpret_cast<const uint4*>(wrow0 + ka));
        const uint4 wb0 = __ldcs(reinterpret_cast<const uint4*>(wrow0 + kb));
        mad_u4(acc0, xa, wa0);
        mad_u4(acc0, xb, wb0);
        if (dual) {
            const uint4 wa1 = __ldcs(reinterpret_cast<const uint4*>(wrow1 + ka));
            const uint4 wb1 = __ldcs(reinterpret_cast<const uint4*>(wrow1 + kb));
            mad_u4(acc1, xa, wa1);
            mad_u4(acc1, xb, wb1);
        }
    }
    for (; k0 < K; k0 += 256) {
        const int k = k0 + lane * 8;
        const uint4 xv = __ldg(reinterpret_cast<const uint4*>(X + k));
        const uint4 wv0 = __ldcs(reinterpret_cast<const uint4*>(wrow0 + k));
        mad_u4(acc0, xv, wv0);
        if (dual) {
            const uint4 wv1 = __ldcs(reinterpret_cast<const uint4*>(wrow1 + k));
            mad_u4(acc1, xv, wv1);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
        if (dual) {
            acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
        }
    }
    if (lane == 0) {
        Out[n0] = __float2bfloat16(acc0);
        if (dual) {
            Out[n1] = __float2bfloat16(acc1);
        }
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
    dim3 grid((unsigned)((N + 15) / 16));
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

# v6c tried unpooler m=1 GEMV; overall 1.069x < v6b 1.071x. Keep off.
_ENABLE_UNPOOLER_GEMV = False


def _get_gemv_mod() -> Any:
    if _load_inline is None:
        return None
    try:
        return _load_inline(
            name="bf16_gemv_v6d",
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
        # m=1：GEMV（unpooler 默认除外，与 v6b 一致；见 _ENABLE_UNPOOLER_GEMV）
        mod = state.get("gemv_mod")
        allow_unpooler = _ENABLE_UNPOOLER_GEMV and layer == "unpooler"
        if (
            mod is not None
            and m == 1
            and k % 256 == 0
            and (layer != "unpooler" or allow_unpooler)
        ):
            try:
                # 传原始 W（[N, K] 行主序），不要传预转置版；前 n 行切片。
                b = state[layer][:n].contiguous()
                out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
                mod.gemv_bf16(x, b, out, n, k)
                return out
            except Exception:
                pass  # 同步检查失败时往下走 v4 路

        # ---- 以下与 v4 / v6b 一字相同 ----
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
