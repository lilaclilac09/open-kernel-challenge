# muse_bf16_proj v5 scout (toward ~1.55×)

Baseline for this note: v4 check **1.057× / 4568 µs**; board **1.055×** (rank ~651). Leaders **~1.55× / ~3119 µs**.

## Legal APIs (judge image `lmsysorg/sglang:dev-muse-spark-open`)

From https://openkernelchallenge.dev/rules: **PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL, CUTLASS**; native CUDA via `torch.utils.cpp_extension.load_inline`. No network installs. CUDA-graph safe `kernel_fn` (no host sync / value-dependent alloc or branch). `prepare()` ≤300s untimed — pack + compile + autotune here.

**This box** (dev sandbox): torch/triton/sglang/tilelang/cutlass **not importable**. Do not treat local import failures as judge truth.

## Why layout plateaued

Profiles on v4 are almost all `nvjet_*` cuBLAS + occasional `splitKreduce`. Wins so far = contiguous pretranspose + hybrid router (m≤7 FP32, else hi/lo) + dispatch for m≥192. Still **1.4–5× above floor** on many shapes (router m=1 FP32 worst; pooler/qkv still leave ~6µs on the table vs SOL).

## 3 fusion bets (ranked)

### 1. Triton split-K GEMM for small-M BF16 (HIGHEST) — **implemented in v5**

- **Targets:** non-router, m∈{1,8,16,32,128} (and tests ≤128); especially **pooler** (n=4096,k=8192) and **qkv_gate** / fat **unpooler**.
- **Why:** cuBLAS launches few CTAs at tiny M; separate splitKreduce adds ~1.7µs. Split-K over K fills ~132 SMs; FP32 atomic then cast BF16.
- **API:** `triton` + `tl.dot` / `tl.atomic_add` (judge).
- **First experiment:** `okc check` of v5 (`TRITON_MAX_M=128`). If < v4, shrink to pooler-only or m≤32, or raise `TRITON_MAX_M` gate off and keep v4 path.
- **Prior miss:** naive GEMV check 8b4524be = **0.87×** (m=1 qkv 0.29×). Must autotune BLOCK_*/SPLIT_K/warps/stages; warmup in `prepare`.

### 2. Fused router hi+lo single kernel (HIGH) — **implemented in v5**

- **Targets:** router m≥8 (scored 8/16/32/128/256); correctness m=9,512.
- **Why:** today 2× `torch.mm` + add ≈14–17µs; x streamed twice. Fuse → one pass, FP32 out (tol 1e-4).
- **API:** Triton (or CuTe later).
- **First experiment:** same v5 check; compare router rows vs v4 profile (~12–13µs GPU).

### 3. CuTe DSL / TileLang / CUTLASS TMA GEMV (MEDIUM, next leap if Triton stalls)

- **Targets:** m=1..32 all layers — approach 4.8 TB/s floors (router/unpooler decode1 still 3–6× SOL).
- **Why:** leaders ~1.55× need near-SOL on decode loads; Hopper TMA + warp specialization beats hand Triton on skinny GEMM.
- **API:** CuTe DSL / TileLang / CUTLASS in image; `prepare` compile budget 300s.
- **First experiment:** one CuTe GEMV for pooler m=1 only behind a shape branch; `okc check`; expand if ≥1.15× on that shape without regressing others.

## Out of scope this note

- Sequential `torch.mm` K-chunks (CTA count unchanged).
- `torch._scaled_mm` (FP8 path, wrong dtype).
- Changing router FP32 threshold away from measured **7**.

## v5 file

`muse_bf16_proj_v5.py` → copied to `submission.py` for check. Fallback: `muse_bf16_proj_v4.py`.
