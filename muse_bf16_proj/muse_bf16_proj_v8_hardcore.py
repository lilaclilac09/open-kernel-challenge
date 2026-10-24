# okc: problem=muse_bf16_proj
"""BF16 projections v8 hardcore：修 load_inline 编译（相对 v7）。

相对 v7（8062e14）的唯一真实改动：给 load_inline 补上 cpp_sources
前向声明。v7 失败根因（log 原文）：

  main.cpp: error: 'gemm_nt_bf16' was not declared in this scope
  main.cpp: error: 'gemm_nt_splitk' was not declared in this scope

load_inline(functions=[...]) 会在单独的 main.cpp 里生成
m.def("name", wrap_pybind_function(name), ...)；那个 TU 只编译
cpp_sources，看不到 cuda_sources 里的定义。v7 / v5 GEMV 都写了
cpp_sources=""，所以 pybind 侧 undeclared，CUDA 本身其实编过了
（8062e14 log 里 nvcc 那步成功）。

本版：
1. cpp_sources = 带签名的前向声明 + #include <torch/extension.h>
2. cuda_sources = 原 v7 kernel + host 实现（一字不改算法）
3. 无 GEMV：热路径只有这一份 CUDA 模块，compile 证明之前不混
4. prepare() 打印 [v8] compiled OK / FAILED + 截断 stderr（进 log_tail）
5. mod is None 时 kernel_fn 与 v4 逐行一致（dispatch m>=192、router 阈值 7）

算法 / 分档 / HARDCORE_MAX_M=128 / split-K 启发式全沿用 v7。
速度 >=1.10 才 scored-submit；本注目标是 profile 出现
gemm_nt_splitk_kernel / gemm_nt_direct_kernel（不再只有 nvjet）。
"""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from torch.utils.cpp_extension import load_inline as _load_inline
except Exception:  # pragma: no cover
    _load_inline = None


# Non-router shapes with m <= this use the hardcore path (scored: 1/8/16/32/128).
# Correctness also hits m=5/7/33/130 — 130 stays on hardcore; 256+ falls to v4.
HARDCORE_MAX_M = 128
# Target in-flight warps on H200 (132 SMs). Warp-per-column grid is (N, SK).
HARDCORE_TARGET_WARPS = 8192
HARDCORE_MIN_K_CHUNK = 256


# Forward decls for the pybind TU. load_inline(functions=...) generates
# main.cpp that only sees cpp_sources — empty string => undeclared (v7).
_CPP_SRC = r"""
#include <torch/extension.h>

void gemm_nt_bf16(
    torch::Tensor A, torch::Tensor B, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K);

void gemm_nt_splitk(
    torch::Tensor A, torch::Tensor B, torch::Tensor Partials, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K, int64_t SK);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Warp-per-column NT GEMM, optional split-K.
//   C[m,n] = sum_k A[m,k] * B[n,k]
// Grid  = (N, SK), Block = 32 (one warp).
// Each lane owns k = k_base + lane*8 + {0..7} inside the K-slice, reads
// uint4 (8 bf16) from A-row and B-row, converts bf162->float2, FMAs into
// a scalar accumulator, then warp-shuffles. Lane 0 writes the partial.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float _dot8_bf16(const uint4 &av, const uint4 &bv) {
    float acc = 0.f;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t xa = (p == 0) ? av.x : (p == 1) ? av.y : (p == 2) ? av.z : av.w;
        const uint32_t wa = (p == 0) ? bv.x : (p == 1) ? bv.y : (p == 2) ? bv.z : bv.w;
        const float2 af = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa));
        const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wa));
        acc += af.x * wf.x + af.y * wf.y;
    }
    return acc;
}

__device__ __forceinline__ float _warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// K-slice [k_start, k_end) must satisfy k_start % 8 == 0; k_end may be any
// (tail handled scalar). Caller rounds chunk starts to 8.
__device__ __forceinline__ float _col_partial(
    const __nv_bfloat16* __restrict__ arow,
    const __nv_bfloat16* __restrict__ brow,
    int k_start, int k_end, int lane)
{
    float acc = 0.f;
    // Vectorized body: 32 lanes * 8 bf16 = 256-wide steps.
    int k0 = k_start;
    for (; k0 + 256 <= k_end; k0 += 256) {
        const int k = k0 + lane * 8;
        const uint4 av = *reinterpret_cast<const uint4*>(arow + k);
        const uint4 bv = *reinterpret_cast<const uint4*>(brow + k);
        acc += _dot8_bf16(av, bv);
    }
    // Tail: fewer than 256 elems left. Still try uint4 per lane when in range.
    for (; k0 < k_end; k0 += 256) {
        const int k = k0 + lane * 8;
        if (k + 8 <= k_end) {
            const uint4 av = *reinterpret_cast<const uint4*>(arow + k);
            const uint4 bv = *reinterpret_cast<const uint4*>(brow + k);
            acc += _dot8_bf16(av, bv);
        } else if (k < k_end) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (k + j < k_end) {
                    acc += __bfloat162float(arow[k + j]) * __bfloat162float(brow[k + j]);
                }
            }
        }
    }
    return _warp_sum(acc);
}

__global__ void __launch_bounds__(32) gemm_nt_splitk_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M,K]
    const __nv_bfloat16* __restrict__ B,   // [N,K]
    float* __restrict__ Partials,          // [SK, M, N]
    int M, int N, int K, int SK)
{
    const int n = (int)blockIdx.x;
    const int sk = (int)blockIdx.y;
    if (n >= N || sk >= SK) return;
    const int lane = (int)threadIdx.x;

    // Evenly sized K-chunks, start aligned to 8 so uint4 loads are aligned
    // (A/B row starts are 16-byte aligned when K % 8 == 0, which Python checks).
    int chunk = (K + SK - 1) / SK;
    chunk = (chunk + 7) & ~7;
    const int k_start = sk * chunk;
    if (k_start >= K) {
        if (lane == 0) {
            for (int m = 0; m < M; ++m) {
                Partials[(size_t)sk * (size_t)M * (size_t)N + (size_t)m * (size_t)N + n] = 0.f;
            }
        }
        return;
    }
    int k_end = k_start + chunk;
    if (k_end > K) k_end = K;

    const __nv_bfloat16* brow = B + (size_t)n * (size_t)K;
    for (int m = 0; m < M; ++m) {
        const __nv_bfloat16* arow = A + (size_t)m * (size_t)K;
        float sum = _col_partial(arow, brow, k_start, k_end, lane);
        if (lane == 0) {
            Partials[(size_t)sk * (size_t)M * (size_t)N + (size_t)m * (size_t)N + n] = sum;
        }
    }
}

// SK == 1 fast path: same math, write bf16 directly (no partials / reduce).
__global__ void __launch_bounds__(32) gemm_nt_direct_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ Out,       // [M,N]
    int M, int N, int K)
{
    const int n = (int)blockIdx.x;
    if (n >= N) return;
    const int lane = (int)threadIdx.x;
    const __nv_bfloat16* brow = B + (size_t)n * (size_t)K;
    for (int m = 0; m < M; ++m) {
        const __nv_bfloat16* arow = A + (size_t)m * (size_t)K;
        float sum = _col_partial(arow, brow, 0, K, lane);
        if (lane == 0) {
            Out[(size_t)m * (size_t)N + n] = __float2bfloat16(sum);
        }
    }
}

__global__ void reduce_splitk_kernel(
    const float* __restrict__ Partials,    // [SK, M, N]
    __nv_bfloat16* __restrict__ Out,       // [M, N]
    int M, int N, int SK)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = M * N;
    if (idx >= total) return;
    float s = 0.f;
    for (int sk = 0; sk < SK; ++sk) {
        s += Partials[(size_t)sk * (size_t)total + idx];
    }
    Out[idx] = __float2bfloat16(s);
}

void gemm_nt_bf16(
    torch::Tensor A, torch::Tensor B, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Out.is_cuda(), "CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16, "bf16");
    TORCH_CHECK(Out.scalar_type() == at::kBFloat16, "out bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Out.is_contiguous(), "contiguous");
    TORCH_CHECK(A.numel() >= M * K && B.numel() >= N * K && Out.numel() >= M * N, "numel");
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_nt_direct_kernel<<<dim3((unsigned)N), dim3(32), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr()),
        (int)M, (int)N, (int)K);
}

void gemm_nt_splitk(
    torch::Tensor A, torch::Tensor B, torch::Tensor Partials, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K, int64_t SK)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Partials.is_cuda() && Out.is_cuda(), "CUDA");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16, "bf16");
    TORCH_CHECK(Partials.scalar_type() == at::kFloat, "partials fp32");
    TORCH_CHECK(Out.scalar_type() == at::kBFloat16, "out bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Partials.is_contiguous() && Out.is_contiguous(), "contiguous");
    TORCH_CHECK(SK >= 1, "SK");
    TORCH_CHECK(Partials.numel() >= SK * M * N && Out.numel() >= M * N, "numel");
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_nt_splitk_kernel<<<dim3((unsigned)N, (unsigned)SK), dim3(32), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<float*>(Partials.data_ptr()),
        (int)M, (int)N, (int)K, (int)SK);
    const int total = (int)(M * N);
    const int block = 256;
    const int grid = (total + block - 1) / block;
    reduce_splitk_kernel<<<dim3((unsigned)grid), dim3((unsigned)block), 0, stream>>>(
        reinterpret_cast<const float*>(Partials.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr()),
        (int)M, (int)N, (int)SK);
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


def _pick_sk(m: int, n: int, k: int) -> int:
    """Choose split-K so N*SK ≈ HARDCORE_TARGET_WARPS, chunk >= MIN_K_CHUNK."""
    # Warp-per-column already launches N warps at SK=1. Split further only when
    # that under-fills the GPU, and never make a K-chunk smaller than 256.
    max_by_k = max(1, k // HARDCORE_MIN_K_CHUNK)
    if n >= HARDCORE_TARGET_WARPS:
        return 1
    sk = (HARDCORE_TARGET_WARPS + n - 1) // n
    return max(1, min(sk, max_by_k))


def _get_hardcore_mod() -> Any:
    if _load_inline is None:
        print("[v8] load_inline unavailable")
        return None
    try:
        mod = _load_inline(
            name="bf16_hardcore_v8",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gemm_nt_bf16", "gemm_nt_splitk"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True,
        )
        print("[v8] hardcore_mod compiled OK")
        print("[v8] exports:", [n for n in ("gemm_nt_bf16", "gemm_nt_splitk") if hasattr(mod, n)])
        return mod
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # Truncate so log_tail still shows the undeclared / ninja tip.
        if len(msg) > 2500:
            msg = msg[:1200] + "\n...[truncated]...\n" + msg[-1200:]
        print(f"[v8] hardcore_mod compile FAILED: {msg}")
        return None


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[f"{name}_t"] = state[name].t().contiguous()
    ut = state["unpooler_t"]
    state["unpooler_t_1024"] = ut[:, :1024].contiguous()
    state["unpooler_t_8192"] = ut[:, :8192].contiguous()
    # Pre-slice row-major W for hardcore (avoids [:n].contiguous() in timed path
    # for the common full-width / unpooler-1024 cases).
    for name in ("qkv_gate", "o", "pooler"):
        state[f"{name}_w"] = state[name]  # already contiguous full [N,K]
    state["unpooler_w_1024"] = state["unpooler"][:1024].contiguous()
    state["unpooler_w_8192"] = state["unpooler"][:8192].contiguous()
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["hardcore_mod"] = _get_hardcore_mod()
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


def _w_row(state: dict[str, Any], layer: str, n: int) -> torch.Tensor:
    """Row-major W[:n] for hardcore."""
    if layer == "unpooler":
        if n == 1024:
            return state["unpooler_w_1024"]
        if n == 8192:
            return state["unpooler_w_8192"]
        return state["unpooler"][:n].contiguous()
    w = state[f"{layer}_w"]
    if n == w.shape[0]:
        return w
    return w[:n].contiguous()


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"].contiguous()
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]
    k = cfg["k"]

    if layer != "router":
        mod = state.get("hardcore_mod")
        if mod is not None and m <= HARDCORE_MAX_M and k % 8 == 0:
            try:
                w = _w_row(state, layer, n)
                out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
                sk = _pick_sk(m, n, k)
                if sk == 1:
                    mod.gemm_nt_bf16(x, w, out, m, n, k)
                else:
                    partials = torch.empty(
                        (sk, m, n), device=x.device, dtype=torch.float32
                    )
                    mod.gemm_nt_splitk(x, w, partials, out, m, n, k, sk)
                return out
            except Exception:
                pass  # sync failure only; async faults surface later

        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
