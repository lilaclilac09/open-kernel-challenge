# muse_bf16_proj A/B results

| ID | Variant | µs | × | Notes |
|---|---|---:|---:|---|
| 0eadad47 | A baseline-like | 4813.5 | 1.004 | cuBLASLt nvjet; router m=8 FP32 ~26µs |
| 0abe9ee3 | B naive Triton all sizes | 8258.6 | 0.585 | correct, slow |
| 8b4524be | C Triton GEMV only m=1 | 5555.0 | 0.870 | m=1 loses to nvjet |
| e4f4865e | D contig transpose + torch.mm; SGLang router | 4734.4 | 1.020 | pooler wins; keep mm |
| 12c92415 | E D + always hi/lo router | 4638.0 | **1.042** | verify1 router 1.82×; decode1 router 0.72× |
| c945f1a2 | E scored submit | (queued) | | same hash as 12c |
| 10ffe3c0 | F hybrid router m<8 FP32 else hi/lo + mm/dispatch split | (queued) | | |

## Takeaways
- Naive Triton GEMV cannot beat H200 nvjet splitK for these shapes.
- Contiguous `W.t()` + `torch.mm` beats `_bf16_gemm_dispatch` on pooler/unpooler mid-m.
- Always hi/lo helps m=8 router a lot but hurts m=1 (~14µs vs ~10µs FP32).
- Next: hybrid router threshold; large-m fallback to dispatch; then CuTe/CUTLASS for small-M.

## Attention
- 78ab10ad: baseline/FA3 submission check (queued)
