# muse_bf16_proj A/B results

| ID | Variant | µs | × | Mode |
|---|---|---:|---:|---|
| 0eadad47 | A baseline-like | 4813.5 | 1.004 | check |
| 0abe9ee3 | B naive Triton | 8258.6 | 0.585 | check |
| 8b4524be | C Triton m=1 only | 5555.0 | 0.870 | check |
| e4f4865e | D contig mm + SGLang router | 4734.4 | 1.020 | check |
| 12c92415 | E always hi/lo | 4638.0 | 1.042 | check |
| c945f1a2 | E scored | 4674.5 | **1.033** | **board** |
| 10ffe3c0 | F hybrid router m≤7 FP32 else hi/lo; mm + dispatch m≥192 | 4598.7 | **1.050** | check |
| c888 / b4cb | v3 (other agent): threshold 8 + always mm | queued | | withdraw b4cb |

## Attention
| 78ab10ad | FA3 baseline | 3781.7 | 0.996 | check |

## Limits (after these)
See `okc problems`. Rank ~655 at 1.033×; top ~1.55×.

## Next
1. Scored-submit F hybrid (1.050 check).
2. Avoid naive Triton; keep graph-safe.
3. Explore CuTe/CUTLASS / better small-M; protect submission.py from cross-agent overwrites.
