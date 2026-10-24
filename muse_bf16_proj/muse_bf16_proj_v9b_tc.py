# okc: problem=muse_bf16_proj
"""BF16 projections v9b：修 v9 tensor-core 正确性（mma.sync fragment 布局）。

相对 v9 check ac2f144（5/9，qkv_gate m=1 rel L2 1.154）：
1. A fragment 寄存器顺序按 PTX m16n8k16 纠正：a1=行+8 同 K，a2=同行列 K+8
   （v9 把这两个对调了）。
2. C/D 写回按 PTX：d0=(g,2t), d1=(g,2t+1), d2=(g+8,2t), d3=(g+8,2t+1)
   （v9 误用近似 m16n8k8 的 (g,t)/(g+8,t)/(g,t+4)/(g+8,t+4)）。
3. TC 只走 m in [8, 128]；m<8（含 m=1）强制 v4 torch.mm / dispatch，
   避免 BM=64 在 tiny-M 上既浪费又难验。
4. prepare() 里加小矩阵自检（有 GPU 时）；无 GPU 跳过，靠 okc。

B .col fragment 本身与 PTX 一致，未改。router / m>=192 仍照抄 v4。
"""

from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from torch.utils.cpp_extension import load_inline as _load_inline
except Exception:  # pragma: no cover
    _load_inline = None


# TC only for m in [TC_MIN_M, TC_MAX_M]. Tiny M (esp. m=1) wastes BM=64 and
# was the first correctness fail in v9; keep those on v4 torch.mm.
TC_MIN_M = 8
TC_MAX_M = 128
# Aim for enough CTAs to keep ~2 waves on 132 SMs when M*N tiles are few.
TC_TARGET_CTAS = 512
TC_MIN_K_CHUNK = 256

# H200 = 132 SMs; tile 64x64 => grid_m*grid_n for full layers.
# qkv N=2304 -> 36 N-tiles; pooler 64; o 128; unpooler 128 (or 16 when n=1024).


_CPP_SRC = r"""
#include <torch/extension.h>

void gemm_nt_tc(
    torch::Tensor A, torch::Tensor B, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K);

void gemm_nt_tc_splitk(
    torch::Tensor A, torch::Tensor B, torch::Tensor Partials, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K, int64_t SK);
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// H200 (sm90) BF16 NT GEMM via mma.sync.m16n8k16 — NOT scalar FMA (that was v8).
//
//   C[m,n] = sum_k A[m,k] * B[n,k]     // A:[M,K] rm, B:[N,K] rm, C:[M,N] rm
//
// Tile: BM=64, BN=64, BK=32. Block = 128 threads = 4 warps in 2x2.
// Each warp owns a 32x32 output patch = 2 (along M) x 4 (along N) MMA 16x8.
//
// Fragment layout: PTX ISA mma.m16n8k16 (fp / bf16) — FIXED in v9b.
//   groupID = lane >> 2;  tid = lane & 3;   (threadID_in_group)
//   A .row  (8 bf16 = 4x u32 / thread), register order matters:
//     a0 pack -> (groupID,   2*tid + {0,1})
//     a1 pack -> (groupID+8, 2*tid + {0,1})       // ROW+8, same K  (v9 had this in a2)
//     a2 pack -> (groupID,   2*tid + {0,1} + 8)   // same row, K+8  (v9 had this in a1)
//     a3 pack -> (groupID+8, 2*tid + {0,1} + 8)
//   B .col  (4 bf16 = 2x u32), into logical B_gemm[K,N] = B_rm^T:
//     b0 pack -> (k=2*tid+{0,1},     n=groupID)
//     b1 pack -> (k=2*tid+{0,1}+8,   n=groupID)
//   D f32 (4 / thread) — NOT the m16n8k8 (g,t)/(g,t+4) layout:
//     d0 -> (groupID,   2*tid)
//     d1 -> (groupID,   2*tid + 1)
//     d2 -> (groupID+8, 2*tid)
//     d3 -> (groupID+8, 2*tid + 1)
// =============================================================================

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int WARP_SIZE = 32;
constexpr int NWARPS = 4;
constexpr int THREADS = NWARPS * WARP_SIZE;  // 128
constexpr int WARP_M = 2;
constexpr int WARP_N = 2;
constexpr int WGM = BM / WARP_M;  // 32
constexpr int WGN = BN / WARP_N;  // 32
// pad K by 8 bf16 to cut smem bank conflicts a bit
constexpr int BKP = BK + 8;

__device__ __forceinline__ uint32_t pack_bf16(const __nv_bfloat16 a,
                                              const __nv_bfloat16 b) {
    uint32_t u;
    // low 16 = a, high 16 = b  (matches .bf16x2 used by mma)
    u = (uint32_t)(__bfloat16_as_ushort(a)) |
        ((uint32_t)(__bfloat16_as_ushort(b)) << 16);
    return u;
}

__device__ __forceinline__ void mma_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Cooperative load of a BM x BK A-tile and BN x BK B-tile into smem.
// A,B are bf16 row-major. Out-of-range rows/cols zero-filled.
__device__ __forceinline__ void load_tiles(
    const __nv_bfloat16* __restrict__ A,  // [M,K]
    const __nv_bfloat16* __restrict__ B,  // [N,K]
    __nv_bfloat16 Asmem[BM][BKP],
    __nv_bfloat16 Bsmem[BN][BKP],
    int M, int N, int K,
    int m0, int n0, int k0)
{
    // 128 threads; each moves 16 bf16 (= one uint4) per step when possible.
    // A tile: BM*BK = 64*32 = 2048 bf16 = 128 * 16 -> exactly 1 step of uint4*2? 
    // 2048/8 = 256 uint4-worth if 8 bf16/thread; with 128 threads -> 2 iters of 8.
    #pragma unroll
    for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
        int row = i / BK;
        int col = i % BK;
        int gm = m0 + row;
        int gk = k0 + col;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (gm < M && gk < K) v = A[(size_t)gm * (size_t)K + gk];
        Asmem[row][col] = v;
    }
    #pragma unroll
    for (int i = threadIdx.x; i < BN * BK; i += THREADS) {
        int row = i / BK;
        int col = i % BK;
        int gn = n0 + row;
        int gk = k0 + col;
        __nv_bfloat16 v = __float2bfloat16(0.f);
        if (gn < N && gk < K) v = B[(size_t)gn * (size_t)K + gk];
        Bsmem[row][col] = v;
    }
}

__device__ __forceinline__ void load_a_frags(
    uint32_t Ra[2][4],  // [mma_m][4 regs]
    const __nv_bfloat16 Asmem[BM][BKP],
    int row0, int k_off, int lane)
{
    const int group = lane >> 2;
    const int tid = lane & 3;
    const int col = tid * 2;
    #pragma unroll
    for (int mm = 0; mm < 2; ++mm) {
        const int r0 = row0 + mm * 16 + group;
        const int r1 = r0 + 8;
        // PTX: a0=(r0,k), a1=(r1,k), a2=(r0,k+8), a3=(r1,k+8)
        Ra[mm][0] = pack_bf16(Asmem[r0][k_off + col],
                              Asmem[r0][k_off + col + 1]);
        Ra[mm][1] = pack_bf16(Asmem[r1][k_off + col],
                              Asmem[r1][k_off + col + 1]);
        Ra[mm][2] = pack_bf16(Asmem[r0][k_off + col + 8],
                              Asmem[r0][k_off + col + 9]);
        Ra[mm][3] = pack_bf16(Asmem[r1][k_off + col + 8],
                              Asmem[r1][k_off + col + 9]);
    }
}

__device__ __forceinline__ void load_b_frags(
    uint32_t Rb[4][2],  // [mma_n][2 regs]
    const __nv_bfloat16 Bsmem[BN][BKP],
    int col0, int k_off, int lane)
{
    const int group = lane >> 2;
    const int tid = lane & 3;
    const int kpair = tid * 2;
    #pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
        // B .col: (k, n=group) with n_base = col0 + nn*8 + group
        const int n = col0 + nn * 8 + group;
        Rb[nn][0] = pack_bf16(Bsmem[n][k_off + kpair],
                              Bsmem[n][k_off + kpair + 1]);
        Rb[nn][1] = pack_bf16(Bsmem[n][k_off + kpair + 8],
                              Bsmem[n][k_off + kpair + 9]);
    }
}

template <bool WritePartials>
__global__ void __launch_bounds__(THREADS)
gemm_nt_tc_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ Partials,          // [SK,M,N] if WritePartials
    __nv_bfloat16* __restrict__ Out,       // [M,N] if !WritePartials
    int M, int N, int K, int SK,
    int k_start, int k_end)
{
    // grid: (ceil(N/BN), ceil(M/BM), SK)  — z is split-K index
    const int n0 = (int)blockIdx.x * BN;
    const int m0 = (int)blockIdx.y * BM;
    const int sk = (int)blockIdx.z;
    if (n0 >= N || m0 >= M) return;

    __shared__ __nv_bfloat16 Asmem[BM][BKP];
    __shared__ __nv_bfloat16 Bsmem[BN][BKP];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_m = warp / WARP_N;       // 0..1
    const int warp_n = warp % WARP_N;       // 0..1
    const int row0 = warp_m * WGM;          // 0 or 32 within tile
    const int col0 = warp_n * WGN;          // 0 or 32

    // Accumulators: 2 (M MMA) x 4 (N MMA) x 4 floats
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[i][j][r] = 0.f;

    // This block's K range (already sliced by host for split-K).
    int ks = k_start;
    int ke = k_end;
    if (ks < 0) ks = 0;
    if (ke > K) ke = K;

    for (int k0 = ks; k0 < ke; k0 += BK) {
        load_tiles(A, B, Asmem, Bsmem, M, N, K, m0, n0, k0);
        __syncthreads();

        // BK=32 = two k16 steps for m16n8k16
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t Ra[2][4];
            uint32_t Rb[4][2];
            load_a_frags(Ra, Asmem, row0, kk, lane);
            load_b_frags(Rb, Bsmem, col0, kk, lane);
            #pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                #pragma unroll
                for (int nn = 0; nn < 4; ++nn) {
                    mma_m16n8k16(
                        acc[mm][nn][0], acc[mm][nn][1],
                        acc[mm][nn][2], acc[mm][nn][3],
                        Ra[mm][0], Ra[mm][1], Ra[mm][2], Ra[mm][3],
                        Rb[nn][0], Rb[nn][1]);
                }
            }
        }
        __syncthreads();
    }

    // Store
    const int group = lane >> 2;
    const int tid = lane & 3;
    #pragma unroll
    for (int mm = 0; mm < 2; ++mm) {
        #pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
            // PTX m16n8k16 D: d0=(g,2t), d1=(g,2t+1), d2=(g+8,2t), d3=(g+8,2t+1)
            const int r_lo = row0 + mm * 16 + group;
            const int r_hi = r_lo + 8;
            const int c_lo = col0 + nn * 8 + tid * 2;
            const int c_hi = c_lo + 1;
            const float vals[4] = {
                acc[mm][nn][0], acc[mm][nn][1],
                acc[mm][nn][2], acc[mm][nn][3]};
            // vals[0]->(r_lo,c_lo), [1]->(r_lo,c_hi), [2]->(r_hi,c_lo), [3]->(r_hi,c_hi)
            const int rr[4] = {r_lo, r_lo, r_hi, r_hi};
            const int cc[4] = {c_lo, c_hi, c_lo, c_hi};
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int gm = m0 + rr[t];
                const int gn = n0 + cc[t];
                if (gm < M && gn < N) {
                    if constexpr (WritePartials) {
                        Partials[(size_t)sk * (size_t)M * (size_t)N
                                 + (size_t)gm * (size_t)N + (size_t)gn] = vals[t];
                    } else {
                        Out[(size_t)gm * (size_t)N + (size_t)gn] =
                            __float2bfloat16(vals[t]);
                    }
                }
            }
        }
    }
}

__global__ void reduce_splitk_kernel(
    const float* __restrict__ Partials,
    __nv_bfloat16* __restrict__ Out,
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

// SK>1 path: dedicated kernel derives [k_start,k_end) from blockIdx.z * chunk.
// (SK==1 uses gemm_nt_tc_kernel<false> launched from gemm_nt_tc.)

__global__ void __launch_bounds__(THREADS)
gemm_nt_tc_splitk_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ Partials,
    int M, int N, int K, int SK, int chunk)
{
    const int n0 = (int)blockIdx.x * BN;
    const int m0 = (int)blockIdx.y * BM;
    const int sk = (int)blockIdx.z;
    if (n0 >= N || m0 >= M || sk >= SK) return;

    int k_start = sk * chunk;
    if (k_start >= K) {
        // Zero this block's partial tile (only threads that would write).
        // Simpler: let accum stay 0 and fall through to store.
    }
    int k_end = k_start + chunk;
    if (k_end > K) k_end = K;

    __shared__ __nv_bfloat16 Asmem[BM][BKP];
    __shared__ __nv_bfloat16 Bsmem[BN][BKP];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_m = warp / WARP_N;
    const int warp_n = warp % WARP_N;
    const int row0 = warp_m * WGM;
    const int col0 = warp_n * WGN;

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[i][j][r] = 0.f;

    for (int k0 = k_start; k0 < k_end; k0 += BK) {
        load_tiles(A, B, Asmem, Bsmem, M, N, K, m0, n0, k0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            if (k0 + kk >= k_end) break;
            uint32_t Ra[2][4];
            uint32_t Rb[4][2];
            load_a_frags(Ra, Asmem, row0, kk, lane);
            load_b_frags(Rb, Bsmem, col0, kk, lane);
            #pragma unroll
            for (int mm = 0; mm < 2; ++mm)
                #pragma unroll
                for (int nn = 0; nn < 4; ++nn)
                    mma_m16n8k16(
                        acc[mm][nn][0], acc[mm][nn][1],
                        acc[mm][nn][2], acc[mm][nn][3],
                        Ra[mm][0], Ra[mm][1], Ra[mm][2], Ra[mm][3],
                        Rb[nn][0], Rb[nn][1]);
        }
        __syncthreads();
    }

    const int group = lane >> 2;
    const int tid = lane & 3;
    #pragma unroll
    for (int mm = 0; mm < 2; ++mm) {
        #pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
            // PTX m16n8k16 D: d0=(g,2t), d1=(g,2t+1), d2=(g+8,2t), d3=(g+8,2t+1)
            const int r_lo = row0 + mm * 16 + group;
            const int r_hi = r_lo + 8;
            const int c_lo = col0 + nn * 8 + tid * 2;
            const int c_hi = c_lo + 1;
            const float vals[4] = {
                acc[mm][nn][0], acc[mm][nn][1],
                acc[mm][nn][2], acc[mm][nn][3]};
            const int rr[4] = {r_lo, r_lo, r_hi, r_hi};
            const int cc[4] = {c_lo, c_hi, c_lo, c_hi};
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int gm = m0 + rr[t];
                const int gn = n0 + cc[t];
                if (gm < M && gn < N) {
                    Partials[(size_t)sk * (size_t)M * (size_t)N
                             + (size_t)gm * (size_t)N + (size_t)gn] = vals[t];
                }
            }
        }
    }
}

void gemm_nt_tc(
    torch::Tensor A, torch::Tensor B, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Out.is_cuda(), "CUDA");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16, "bf16");
    TORCH_CHECK(Out.scalar_type() == at::kBFloat16, "out bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Out.is_contiguous(), "contiguous");
    TORCH_CHECK(A.numel() >= M * K && B.numel() >= N * K && Out.numel() >= M * N, "numel");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(THREADS);
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM), 1);
    gemm_nt_tc_kernel<false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        nullptr,
        reinterpret_cast<__nv_bfloat16*>(Out.data_ptr()),
        (int)M, (int)N, (int)K, 1, 0, (int)K);
}

void gemm_nt_tc_splitk(
    torch::Tensor A, torch::Tensor B, torch::Tensor Partials, torch::Tensor Out,
    int64_t M, int64_t N, int64_t K, int64_t SK)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && Partials.is_cuda() && Out.is_cuda(), "CUDA");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16, "bf16");
    TORCH_CHECK(Partials.scalar_type() == at::kFloat, "partials fp32");
    TORCH_CHECK(Out.scalar_type() == at::kBFloat16, "out bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() &&
                Partials.is_contiguous() && Out.is_contiguous(), "contiguous");
    TORCH_CHECK(SK >= 1, "SK");
    TORCH_CHECK(Partials.numel() >= SK * M * N && Out.numel() >= M * N, "numel");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(THREADS);
    dim3 grid((unsigned)((N + BN - 1) / BN),
              (unsigned)((M + BM - 1) / BM),
              (unsigned)SK);
    int chunk = (int)((K + SK - 1) / SK);
    chunk = (chunk + BK - 1) / BK * BK;
    if (chunk < BK) chunk = BK;
    gemm_nt_tc_splitk_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<float*>(Partials.data_ptr()),
        (int)M, (int)N, (int)K, (int)SK, chunk);
    const int total = (int)(M * N);
    const int rblock = 256;
    const int rgrid = (total + rblock - 1) / rblock;
    reduce_splitk_kernel<<<dim3((unsigned)rgrid), dim3((unsigned)rblock), 0, stream>>>(
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
    """Split-K so grid_m * grid_n * SK ≈ TC_TARGET_CTAS, chunk >= MIN_K_CHUNK."""
    bm, bn = 64, 64
    tiles = ((m + bm - 1) // bm) * ((n + bn - 1) // bn)
    if tiles <= 0:
        return 1
    max_by_k = max(1, k // TC_MIN_K_CHUNK)
    if tiles >= TC_TARGET_CTAS:
        return 1
    sk = (TC_TARGET_CTAS + tiles - 1) // tiles
    return max(1, min(sk, max_by_k))


def _get_tc_mod() -> Any:
    if _load_inline is None:
        print("[v9b] load_inline unavailable")
        return None
    try:
        mod = _load_inline(
            name="bf16_tc_v9b",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gemm_nt_tc", "gemm_nt_tc_splitk"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=True,
        )
        print("[v9b] tc_mod compiled OK")
        print(
            "[v9b] exports:",
            [n for n in ("gemm_nt_tc", "gemm_nt_tc_splitk") if hasattr(mod, n)],
        )
        return mod
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if len(msg) > 2500:
            msg = msg[:1200] + "\n...[truncated]...\n" + msg[-1200:]
        print(f"[v9b] tc_mod compile FAILED: {msg}")
        return None



def _self_check_tc(mod: Any) -> bool:
    """Numerical smoke vs torch.mm on small NT GEMMs. Skip if no CUDA/GPU.

    Returns False if any relative L2 exceeds 1e-2 (same order as okc tol);
    caller should disable the TC path. Box often has no GPU — then skip.
    """
    if mod is None:
        return False
    try:
        if not torch.cuda.is_available():
            print("[v9b] self-check skipped (no CUDA on this host)")
            return True  # allow submit; okc has GPU
    except Exception:
        print("[v9b] self-check skipped (cuda probe failed)")
        return True
    device = torch.device("cuda")
    shapes = [(8, 64, 64), (16, 128, 128), (32, 64, 256), (8, 128, 256)]
    try:
        for m, n, k in shapes:
            a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
            b = torch.randn((n, k), device=device, dtype=torch.bfloat16)
            out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            mod.gemm_nt_tc(a, b, out, m, n, k)
            torch.cuda.synchronize()
            ref = torch.mm(a, b.t())
            diff = (out.float() - ref.float()).norm().item()
            denom = max(ref.float().norm().item(), 1e-8)
            rel = diff / denom
            if rel > 0.01:
                print(f"[v9b] self-check FAIL m={m} n={n} k={k} rel_L2={rel:.4f}")
                return False
            print(f"[v9b] self-check OK m={m} n={n} k={k} rel_L2={rel:.6f}")
        return True
    except Exception as e:
        print(f"[v9b] self-check ERROR: {type(e).__name__}: {e}")
        return False


def _warmup(mod: Any, state: dict[str, Any]) -> None:
    """Untimed smoke: one launch per layer at a couple of m's so JIT/caches settle."""
    if mod is None:
        return
    device = state["qkv_gate"].device
    shapes = [
        ("qkv_gate", 32, 2304, 8192),
        ("o", 16, 8192, 1024),
        ("pooler", 8, 4096, 8192),
        ("unpooler", 128, 8192, 4096),
    ]
    try:
        for layer, m, n, k in shapes:
            x = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            w = _w_row(state, layer, n)
            out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            sk = _pick_sk(m, n, k)
            if sk == 1:
                mod.gemm_nt_tc(x, w, out, m, n, k)
            else:
                partials = torch.empty((sk, m, n), device=device, dtype=torch.float32)
                mod.gemm_nt_tc_splitk(x, w, partials, out, m, n, k, sk)
        torch.cuda.synchronize()
        print("[v9b] warmup OK")
    except Exception as e:
        print(f"[v9b] warmup FAILED: {type(e).__name__}: {e}")
        # Leave mod in place; kernel_fn will try/except per call.


def prepare(weights: Weights) -> dict[str, Any]:
    state: dict[str, Any] = {name: w.contiguous() for name, w in weights.items()}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        state[f"{name}_t"] = state[name].t().contiguous()
    ut = state["unpooler_t"]
    state["unpooler_t_1024"] = ut[:, :1024].contiguous()
    state["unpooler_t_8192"] = ut[:, :8192].contiguous()
    for name in ("qkv_gate", "o", "pooler"):
        state[f"{name}_w"] = state[name]
    state["unpooler_w_1024"] = state["unpooler"][:1024].contiguous()
    state["unpooler_w_8192"] = state["unpooler"][:8192].contiguous()
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    mod = _get_tc_mod()
    if mod is not None and not _self_check_tc(mod):
        print("[v9b] disabling tc_mod after self-check failure")
        mod = None
    state["tc_mod"] = mod
    _warmup(mod, state)
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
        mod = state.get("tc_mod")
        if mod is not None and TC_MIN_M <= m <= TC_MAX_M and k % 16 == 0:
            try:
                w = _w_row(state, layer, n)
                out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
                sk = _pick_sk(m, n, k)
                if sk == 1:
                    mod.gemm_nt_tc(x, w, out, m, n, k)
                else:
                    partials = torch.empty(
                        (sk, m, n), device=x.device, dtype=torch.float32
                    )
                    mod.gemm_nt_tc_splitk(x, w, partials, out, m, n, k, sk)
                return out
            except Exception:
                pass

        if m >= 192:
            return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
        return torch.mm(x, _w_t(state, layer, n))

    if m <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
