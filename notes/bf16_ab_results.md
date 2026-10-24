# muse_bf16_proj A/B

## A — baseline-like (job 0eadad47a2dc4fdd, check)
- Code: SGLang `_bf16_gemm_dispatch_impl` + router FP32 for m<=8 / hi+lo else
- passed=True, speedup≈1.0036 (CI ~0.99–1.02), score_us≈4813.5, baseline_us=4830.8
- checks used: 1

## B — Triton GEMV (m==1) + Triton small-M GEMM (m<=32) + always hi/lo router (job 0abe9ee3d7ba4d8a, check)
- passed=True (correct), speedup≈0.585 (SLOWER), score_us≈8258.6
- Naive Triton tiled GEMM loses badly to cuBLAS on H200 for m=8..32 under CUDA graphs + cold L2.
- checks used: 2

## C — next (queued): GEMV only for m==1, else exact baseline paths
