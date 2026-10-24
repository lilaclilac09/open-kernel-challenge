# okc: problem=muse_bf16_proj
"""BF16 projections v7 hardcore：m<=128 手写 CUDA split-K GEMM。

相对 v4（板上 1.055×）只在非 router、m<=128 加一条路；router / m>=192
dispatch / 其余 torch.mm 与 v4 逐行一致（router 阈值仍是 7）。

设计（对着代码验）：
1. 计算 C[M,N] = A[M,K] @ W[N,K].T，A/W 都是 bf16 行主序。
2. 主 kernel：一个 warp 负责一列 n、一个 K-slice（grid = (N, SK)，
   block = 32）。32 个 lane 沿 K 向量化分工（每次 uint4 = 8 bf16），
   warp shuffle 归约后写入 FP32 partials[sk, m, n]。对每个 m 循环，
   所以 m=1..128 都能走同一条路，无 16 对齐要求（正确性测例里的
   m=5/7/33/130 也安全；m=130 仍走本路，m=256/512/1000 回 v4）。
3. Split-K：目标约 8k warps 填满 H200 的 132 SM；SK==1 时主 kernel
   直接写 bf16 out，跳过 partials + reduce。
4. 第二 kernel：按 (m,n) 把 SK 个 FP32 partial 加起来，写 bf16 out。
5. 不用 mma.sync / wgmma / Triton / CUTLASS —— 上一发 GEMV（v5）在
   profile 里根本没出现自定义 kernel（只有 nvjet），说明要么
   load_inline 失败要么立刻回退；本版先用可证正确的 FMA + 向量化
   load 把 compile→launch→graph 路径跑通。若 check 里看到
   gemm_nt_splitk_kernel，再考虑换成 mma.sync.m16n8k16 的 64x64 tile。
6. prepare() 里 load_inline 编译一次；失败则 hardcore_mod=None，
   kernel_fn 自动退回 v4。print 编译结果到 log_tail 方便诊断。
7. CUDA graph：kernel_fn 无 host sync；partials/out 用 torch.empty
   在 timed 路径分配（与 GEMV / torch.mm 相同约定）。

为何可能赢（赌点）：
- 小 m 上 cuBLAS nvjet 的 tile 在 M 维大量空转；warp-per-column
  按 N 铺满 SM，K 向 split 再加一波并行，理论上更贴带宽。
- 风险：nvjet 已经是 sm90 专用 + 自带 splitK（见 46d5c1 profile），
  手写 FMA 很可能仍然更慢。本注目标是（a）验证 load_inline 可用
  （b）拿到自定义 kernel 的 profile 行（c）正确性过 9/9。
  速度 >=1.10 才 scored-submit。

依赖：torch.utils.cpp_extension.load_inline（需镜像有 nvcc+ninja）。
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
        print("[v7] load_inline unavailable")
        return None
    try:
        mod = _load_inline(
            name="bf16_hardcore_v7",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            functions=["gemm_nt_bf16", "gemm_nt_splitk"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True,
        )
        print("[v7] hardcore_mod compiled OK")
        return mod
    except Exception as e:
        print(f"[v7] hardcore_mod compile FAILED: {type(e).__name__}: {e}")
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
