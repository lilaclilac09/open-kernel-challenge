# okc: problem=muse_bf16_proj
"""BF16 projections v6：全档位重写（m=1 / 8 / 16 / 32 / 128 / 256 每一档都有自己的路）。

和 v4 / v5 / hardcore 的关系（不要混着看）：
- v4（最好实测 1.042x，4599µs）：非 router 一律 torch.mm，m>=192 才走 SGLang
  dispatch；router 阈值 7。v6 保留它的两个已验证赢面：m>=192 的 dispatch、
  router 阈值 7，一字不动。
- v5：只赌 m=1 的 warp-per-output GEMV，其余照抄 v4。v6 把 v5 的 GEMV
  吸收进来当 m=1 的一支（unpooler 除外，见下），但不再只赌 m=1。
- hardcore：m<=128 的手写 CUDA split-K mma tile GEMM + FP32 partials +
  单独归约。v6 不重复它：小 m / 中 m 的 split-K 改用纯 torch 的
  batched matmul（torch.bmm）实现，不写新 CUDA kernel、不依赖 mma 布局，
  正确性由 cuBLAS 背书，风险面完全不同。
- 纯 Triton v2 用户已实测更慢，本版不碰 Triton。

核心洞察（为什么每档不能同一条路）：
- 全部 5 层都是权重-bound：权重一共约 188MB（qkv 38MB、o 17MB、pooler
  67MB、unpooler 67MB），m 越小，算量越小、权重读一遍的占比越高，
  tile 排布的浪费就越致命。按 64x128 tile 估算，一个 GEMM 的 block 数
  约为 ceil(m/64) * ceil(N/128)：m<=32 时 M 方向永远只有 1 个 tile 高，
  block 数全靠 N 撑，qkv 只有 18 个、unpooler 切片后只有 8 个，132 个 SM
  大半空转——这是小 m 的主矛盾。m=128 时 M 方向 2 个 tile，o / unpooler
  已有 128 个 block，不缺并行度；m=256 时 dispatch 已证明是最优解。
- GEMV 不能推广到 m>1：warp-per-output 每个输出都要完整读一遍权重，
  m=1 时权重恰好读一次，m=16 时就要读 16 遍（qkv / pooler 权重 38-67MB，
  远超 H200 约 50MB L2，L2 兜不住），所以 GEMV 只留在 m=1。

按档位的设计表（非 router 层；router 见表后）：

- m=1：qkv_gate / o / pooler 走 v5 GEMV（一个 warp 算一个输出，32 lane
  沿 K 分工、uint4 读、warp shuffle 归约，FP32 累加；block 数 288 /
  1024 / 512）。unpooler 不走 GEMV：它在 m=1 时 N 只有 1024，GEMV 只
  有 128 个 block、每 block K=4096 的长循环，latency-bound，收益最小；
  改走下面的 bmm split-K（S=16，128 个 tile 实例、每实例 K=256 的短循环）。
  若 GEMV 编译失败，qkv / o / pooler 自动退到 bmm split-K，不炸提交。
  下一注：若 check 显示 GEMV 赢、unpooler bmm 也赢，再把 unpooler 也
  换成 GEMV 单独测一次；若 GEMV 输，直接关掉 GEMV 分支。
- m=8：全部非 router 层走 bmm split-K（新，纯 torch）。S 按“把 tile 数
  补到约 128”算：qkv S=8（18->144）、pooler S=4（32->128）、o S=2
  （64->128）、unpooler 切片 S=16（8->128，K/S=256）。实现见
  _bmm_splitk：权重预转置版 [K,N] 按 K 切成 S 段就是 [S,K/S,N] 的 view，
  不拷贝；x 切成 [S,m,K/S] 只拷一份小 x；一次 bmm 出 S 份 partial，
  转 FP32 求和再转回 bf16。为什么不用 hardcore 的 mma split-K：
  bmm 的 partial 是 cuBLAS 内部 FP32 累加后舍入到 bf16、再 FP32 求和，
  精度风险只剩每份一次 bf16 舍入（约 0.4% 量级），远低于 1% 容差，
  且不需要 nvcc、不赌 mma 下标。最大风险：bmm + 归约是 2-3 个 kernel，
  若 bmm 自身 tile 选得不好或归约开销吃掉并行收益，可能反而不如
  torch.mm——check 若显示某层变慢，先把该层的 S 减半，再不行就把该层
  退回 torch.mm（只改 _split_s 的一处表，不要动别的层）。
  Router m=8：hi+lo（铁律，实测约 14µs，FP32 约 26µs，阈值 7 不动）。
- m=16：和 m=8 完全相同的 bmm split-K 与 S（M 方向仍是 1 个 tile 高，
  tile 数、K/S、并行缺口都和 m=8 一样；区别只在每 tile 多算 2 倍行数，
  不影响 S 的选择）。不走 GEMV 的理由同上（权重会被重复读 16 遍）。
  Router m=16：hi+lo torch.mm（和 v4 一样；router 权重只有 8MB、N 只
  有 256，tile 数本来就只有 2 个，split-K 的归约相对开销更大，暂不动）。
  下一注：若 m=8 / 16 的 bmm 赢了，再试把 hi+lo 两路拼成一个 batch=2
  的 bmm（一次 launch 算 hi 和 lo），router 小 m 可能还有一笔。
- m=32：bmm split-K，S 的算法和 m=8 / 16 相同，但 unpooler 已是满
  N=8192（m>8 不切片）：tile 数 64 -> S=2（K/S=2048）。其余三层 S
  不变（qkv 8、pooler 4、o 2）。Router m=32：hi+lo torch.mm，同 m=16
  的理由。下一注：若 bmm 在 32 赢得比 8 / 16 少，说明 tile 内行数上
  来后并行缺口在缩小，32 档可以先退回 torch.mm 保底。
- m=128：只在缺并行度的地方 split-K。qkv tile 数 36 -> S=4（144）、
  pooler 64 -> S=2（128）；o / unpooler 满 N 时 tile 数已是 128，S=1
  走 torch.mm（再 split 只会多一次归约，白交开销）。Router m=128：
  hi+lo torch.mm（每路 tile 数 2x2=4、K=8192 的长循环，单 block 工作量
  大，latency 不是主矛盾；权重小，split 收益有限，暂不动）。
  下一注：128 档若 bmm 赢，下一步是 persistent kernel 而不是更大的 S。
- m=256：非 router 走 SGLang dispatch（v4 已验证的赢面）。v3 全走
  torch.mm 时总分从 1.042x 掉到 1.022x、慢约 128µs，证明大 m dispatch
  有真本事，本版不砍、不替换。Router m=256：hi+lo torch.mm（同上，
  每路 tile 数 4x2=8、K 长循环，暂不动）。

沿用 v4 已验证的赢面（新代码里原样保留）：
- prepare：全部权重 contiguous + 预转置 + unpooler 预切片
  （unpooler_t_1024 / unpooler_t_8192）+ router hi/lo 预转置。
- kernel_fn 开头 x.contiguous()（已连续就是 no-op）。
- m>=192 dispatch 调用形式和 v4 完全相同。
- router 阈值 ROUTER_FP32_MAX_TOKENS = 7（m<=7 走 FP32 F.linear，
  m>=8 走 hi+lo；m=8 的 14µs vs 26µs 是实测铁律）。

新写的东西（本版独有）：
- _split_s(m, n, k)：只用 config 里的 m / n / k 算 S，目标 tile 数约
  128，上限 16 且保证每段 K/S >= 256 且 S 整除 K。64x128 tile 只是
  估算 S 的启发式——即使 cuBLAS 实际 tile 不同，S 倍增 block 数的方向
  仍然对；若 check 显示某档变慢，调 S，不动结构。
- _bmm_splitk(x, wt, s)：纯 torch 的 split-K，见 m=8 的说明。wt 必须
  是 _w_t 返回的 contiguous 预转置版（题面全部 (层, n) 组合都满足，
  代码里仍留 .contiguous() 兜底）。
- m=1 的 GEMV：吸收 v5，unpooler 除外；编译失败则退 bmm / torch.mm。

依赖与风险：
- SGLang dispatch 与 v4 相同，已验证。
- GEMV 走 load_inline，需评分镜像有 nvcc + ninja；失败则 gemv_mod=None，
  自动退回 bmm / v4 路，不炸提交。
- bmm + 归约是 2-3 个 kernel，可能反而不如 torch.mm——check 若变慢，
  先调 S 再退 torch.mm。
- 真机正确性（cuBLAS tile、bf16 partial 舍入、CUDA graph、GEMV nvcc）
  以 okc check 为准。
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
            name="bf16_gemv_v6",
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



def _split_s(m: int, n: int, k: int) -> int:
    """只用 config 的 m / n / k 算 bmm split-K 的段数 S。

    启发式：按 64x128 tile 估算单 GEMM 的 tile 数，把它补到约 128 个；
    上限 16，且保证每段 K/S >= 256 且 S 整除 K（题面 K 都是 2 的幂，
    实际 S 都会落在 2 / 4 / 8 / 16）。tile 数已 >= 128 时返回 1，即不用
    split-K、直接走 torch.mm。返回 1 也覆盖“无法整除 K”的退化情况。
    """
    tiles = ((m + 63) // 64) * ((n + 127) // 128)
    if tiles >= 128:
        return 1
    s = (128 + tiles - 1) // tiles
    max_s = k // 256
    if s > max_s:
        s = max_s
    if s > 16:
        s = 16
    while s > 1 and k % s != 0:
        s -= 1
    return s if s > 1 else 1


def _bmm_splitk(x: torch.Tensor, wt: torch.Tensor, s: int) -> torch.Tensor:
    """纯 torch 的 split-K：把 K 切成 s 段，一次 bmm 并行算 s 份 partial。

    x: [m, k] contiguous；wt: [k, n] contiguous（预转置权重）。
    A = x.view(m, s, k/s).permute(1, 0, 2).contiguous() -> [s, m, k/s]
    （只拷一份小 x，m*k 至多 256*8192 个元素）；
    B = wt.view(s, k/s, n)，这是按 K 切段的 view，不拷贝（wt 行主序，
    前 k/s 行恰好是第一段，依此类推）。
    partials: [s, m, n] bf16（cuBLAS 内部 FP32 累加后舍入到 bf16），
    再转 FP32 求和、转回 bf16。整个过程只有形状分支，没有数据分支。
    """
    m, k = x.shape
    ks = k // s
    a = x.view(m, s, ks).permute(1, 0, 2).contiguous()
    b = wt.view(s, ks, wt.shape[1])
    partials = torch.bmm(a, b)
    return partials.float().sum(dim=0).to(torch.bfloat16)


def kernel_fn(data: Inputs, state: dict[str, Any]) -> torch.Tensor:
    x = data["x"].contiguous()
    cfg = data["config"]
    layer = cfg["layer"]
    n = cfg["n"]
    m = cfg["m"]
    k = cfg["k"]

    if layer != "router":
        # ---- m=1：v5 GEMV（unpooler 除外，见头注释）----
        mod = state.get("gemv_mod")
        if mod is not None and m == 1 and k % 256 == 0 and layer != "unpooler":
            try:
                # 传原始 W（[N, K] 行主序），不要传预转置版；前 n 行切片。
                b = state[layer][:n].contiguous()
                out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
                mod.gemv_bf16(x, b, out, n, k)
                return out
            except Exception:
                pass  # 同步检查失败时往下走 bmm / torch.mm；异步错误不在此列

        # ---- m>=192：v4 已验证的 SGLang dispatch，一字不动 ----
        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)

        # ---- m<192：bmm split-K（S>1）或 torch.mm（S==1）----
        wt = _w_t(state, layer, n)
        s = _split_s(m, n, k)
        if s > 1:
            try:
                # .contiguous() 对题面全部 (层, n) 组合是 no-op（_w_t 返回
                # 的已是 contiguous 预转置 / 预切片版），只防 off-spec 的 n。
                return _bmm_splitk(x, wt.contiguous(), s)
            except Exception:
                pass  # 同步失败时退回 torch.mm；异步错误不在此列
        return torch.mm(x, wt)

    # router 和 v4 逐行一致，阈值 7 不要动（m=8 走 hi+lo，约 14µs）。
    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
