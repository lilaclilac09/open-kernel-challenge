# muse_bf16_proj A/B/C/D

## A — baseline-like (0eadad47a2dc4fdd)
passed=True speedup≈1.0036 CI[0.99,1.02] score_us≈4813.5
Profile: cuBLASLt nvjet splitK already used; router m=8 FP32 path ~26µs (sol 1.8) is the worst gap.

## B — Triton GEMV m==1 + Triton GEMM m<=32 + always hi/lo (0abe9ee3d7ba4d8a)
passed=True speedup≈0.585 score_us≈8259 — correct but much slower.

## C — Triton GEMV only m==1 else baseline (8b4524be41d64ebf)
passed=True speedup≈0.870 — m=1 shapes ~0.3–0.7× (naive Triton GEMV loses to nvjet).

## D — torch.mm on prepare()-transposed contiguous [K,N] (e4f4865ee296404b)
passed=True speedup≈1.020 CI[1.005,1.036] score_us≈4734 — slight gain, still <1.05; do not scored-submit yet.

## E — next: D + always router hi/lo (even m<=8) to attack 26µs FP32 path
