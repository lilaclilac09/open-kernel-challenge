# okc: problem=muse_bf16_proj
"""BF16 projections v5：并行于 v4 的另一条路——只赌 m=1 的专用 GEMV。

和 v4 / hardcore 的区别（不要混着看）：
- v4：非 router 一律 torch.mm（m>=192 才走 SGLang dispatch），m=1 也走 cuBLAS GEMM。
- hardcore：m<=128 走手写 CUDA split-K GEMM（mma.sync 64x64 tile + FP32
  partials + 单独归约 kernel），大 m 走 torch.mm。
- v5：只对 m=1 动刀，其他所有档位、所有层原样照抄 v4。
  m=1 时走一个手写 CUDA GEMV kernel（下面详述）；m!=1 时的 dispatch /
  torch.mm / router 路径和 v4 逐行一致，包括 router 阈值 7（m=8 走 hi+lo）。

针对哪几档 m / 哪几层：
- 只针对：非 router 层（qkv_gate / o / pooler / unpooler），且 m == 1。
- 不针对：router（FP32，仍走 v4 的 F.linear / hi+lo 路）；m = 8 / 16 / 32 /
  128 / 256（仍走 v4）；unpooler m=1 时 n=1024（仍是前 1024 行切片）。

为什么理论上能赢（只针对 m=1 的道理）：
- m=1 是纯带宽活：输出只有一个行向量，算量是 N*K 次乘加，权重却要
  完整读一遍（qkv_gate 约 38MB、pooler 约 67MB、unpooler m=1 时约 8MB）。
  cuBLAS GEMM 是按 GEMM tile 排布的，m=1 时 tile 的 63/64 行是空的，
  占不满 132 个 SM，也很难把权重流式读满带宽——这是 v4 在小 m 上最可能
  漏掉的部分，也是榜首 3.20ms vs v4 4.60ms 的缺口最可能藏的地方。
- 本 kernel 的打法和 GEMM tile 完全不同：一个 warp 算一个输出元素
  （不是一个 tile），32 个 lane 沿 K 方向分工，每个 lane 一次读 8 个
  bf16（一个 uint4，16 字节），warp 内 256 个元素一格往前扫，扫完
  warp shuffle 归约一次。权重每个元素恰好被读一次，lane 的读地址
  连续成 512 字节一段，合并访存好；x 只有 2~16KB，塞得进 L2，
  所有 block 反复读它基本走 L2 不走 DRAM。
- block 数 = ceil(N/8)：qkv_gate 288 个、o 1024 个、pooler 512 个，
  都远超 132 个 SM；unpooler 128 个，刚好约一波（这是它的短板，见风险）。

设计细节（对着代码验，出问题先查这里）：
- 不用 mma、不 split-K、不用 shared memory、不用 Triton、不用 CUTLASS。
  没有 partials 张量、没有第二归约 kernel：warp 内部 shuffle 就是归约。
- 向量化：lane 一次读 uint4（8 个 bf16），按 bf162 成对转 FP32 再乘加，
  FP32 累加，1% 容差下绰绰有余。成对内部的两个元素即使按低/高位
  互换理解，乘积和也不变（a0*b0 + a1*b1 换序还是同一组乘积）。
- 对齐：K 取值只有 1024 / 4096 / 8192，都是 256 的倍数，也是 8 的
  倍数；权重行首、uint4 读地址都 16 字节对齐。Python 侧要求 k % 256 == 0
  才走本 kernel，否则回 v4 路。
- 计时安全：kernel_fn 里只有按 config（m / 层 / n / k）走的分支，没有
  .item() / .cpu() / 同步 / 打印；CUDA 编译只在 prepare（不计时）做一次。

依赖与“需 okc check 验证可用性”的部分：
- SGLang dispatch 那行和 v4 完全相同，是已在最好版里跑过的依赖，不新
  增风险。
- 需 okc check 验证可用性：本 GEMV kernel 走 torch.utils.cpp_extension
  的 load_inline，需要评分镜像里有 nvcc + ninja，且 prepare 里一次编译
  要在 300 秒内完成。编译失败时 prepare 返回的 gemv_mod 为 None，
  kernel_fn 自动退回 v4 路（不会炸掉整个提交），但那时 v5 就等价于 v4。
- 另需 okc check 验证：扩展函数在当前 stream 上 launch、以及 timed
  调用里 torch.empty((1, n)) 的分配，能否被 harness 的 CUDA graph
  重放正确捕获。torch.mm 内部同样要分配输出，hardcore 也在 timed
  调用里 torch.empty 过 partials / out，所以这个假设有先例，但仍是
  本版第一优先要验的正确性项，不是已证实的事。

最大的风险点（按危险程度排）：
1. load_inline 在评分镜像里不可用（无 nvcc / 编译超时）——后果可控：
   退回 v4，等于没交 v5；不会错，只会白交一次。
2. CUDA graph 捕获问题（见上）——后果不可控：若 check 正确性不过，
   先把 GEMV 分支关掉（或只关有问题的层），不要先调速度。
3. m=1 的 cuBLAS 其实已经接近带宽上限——那 GEMV 赢不了多少，
   几何平均里 m=1 只占 6 档 m 中的 1 档，整体收益有限（见文件外说明）。
   这是最可能的“白忙”结局，不是正确性风险。
4. unpooler m=1 只有 128 个 block、每 block 活又小（K=4096），
   可能 latency-bound，收益最小，甚至可能比 torch.mm 慢。
   若 check 里单层拆分显示 unpooler m=1 拖后腿，可把 unpooler
   排除在 GEMV 分支外（只改 kernel_fn 里一处条件）。
5. kernel 已提交后的异步错误无法被 Python 的 try/except 抓住
   （和 hardcore 同一个 caveat）：同步检查失败才回退，launch 后的
   非法访存会在之后才暴露。下标已用纯 Python 模拟验过（见
   test_v5_gemv.py），但真机正确性仍以 okc check 为准。
"""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from torch.utils.cpp_extension import load_inline as _load_inline
except Exception:  # pragma: no cover - torch without cpp_extension
    _load_inline = None

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
            name="bf16_gemv_v5",
            cpp_sources="",
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
        # v5 唯一的新分支：m == 1 的专用 GEMV。其他条件一个字都不动。
        mod = state.get("gemv_mod")
        if mod is not None and m == 1 and k % 256 == 0:
            try:
                # 传原始 W（[N, K] 行主序），不要传预转置版；前 n 行切片。
                b = state[layer][:n].contiguous()
                out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
                mod.gemv_bf16(x, b, out, n, k)
                return out
            except Exception:
                pass  # 同步检查失败时退回 v4 路；kernel 已提交后的异步错误不在此列

        # 以下和 v4 逐行一致。
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    # router 和 v4 逐行一致，阈值 7 不要动（m=8 走 hi+lo，约 14µs）。
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
