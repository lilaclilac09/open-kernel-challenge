# muse_bf16_proj v9 notes — what it takes to beat 1.05×

Board baseline: **v4 scored 1.055× / 4579 µs** (`1d982289`). Leaders ~**1.55× / ~3.12 ms**.
v8 hardcore (scalar FMA, compile fixed): **0.192×** — custom kernels ran, but ~5–60× slower than `nvjet` on m≥8.

## Why v8 lost (do not repeat)

| failure | evidence |
|---|---|
| No tensor cores | profile: `gemm_nt_splitk_kernel` only; m=128 qkv **1131 µs** vs SGLang **17 µs** |
| Warp-per-column FMA | one warp owns one N column, walks full K in registers; A rows re-fetched per m |
| No smem tile reuse | bandwidth + math both lose to cuBLAS/nvjet sm90 tiles |

H200 BF16 tensor peak ~989 TFLOP/s; scalar FMA is the wrong unit of work.

## v4 vs SOL (from `profiles/submission.1d98228907bf4a96.txt`)

`yours − sol` is the µs left on the table per shape (not load-weighted). Bigger gaps first:

| shape | yours | SOL | gap | yours/SOL |
|---|---:|---:|---:|---:|
| router m=256 | 17.7 | 2.7 | 15.0 | 6.6× |
| router m=128 | 16.7 | 2.2 | 14.5 | 7.6× |
| pooler m=256 | 31.1 | 17.4 | 13.7 | 1.8× |
| router m=32/16/8 | ~15 | ~1.8 | ~13 | ~8× |
| qkv m=256/128 | ~18–20 | ~8–10 | ~9–10 | ~2.0× |
| pooler m=1..128 | ~20–23 | ~14 | ~6–8 | ~1.5–1.6× |
| qkv m=1..32 | ~14 | ~8 | ~6 | ~1.7× |
| o (all m) | ~6–9 | ~3.5–4.5 | ~3–5 | ~1.8–2.1× |
| unpooler m=1/8 | ~7 | ~1.8 | ~5 | ~4× (latency) |

Rough read:

1. **Router is the largest absolute gap** but v4 already won m=8 (1.80× via hi+lo). Further router wins need a fused hi+lo GEMM / better small-N path — **out of scope for v9 TC** (v9 keeps v4 router).
2. **Non-router weight-bound layers sit at ~50–70% of SOL.** Closing half that gap on qkv/pooler/o/unpooler across m=1..128 is the realistic path from 1.05→~1.2–1.3. Getting to **1.55×** almost certainly needs **wgmma + TMA-class** bandwidth (what `nvjet_sm90_*` is), not a first mma.sync tile.
3. **m=256 already uses SGLang dispatch** in v4; pooler m=256 is the weak spot (0.99×). v9 leaves m≥192 on dispatch.

## What v9 is (draft only — no check yet)

File: `muse_bf16_proj_v9_tc.py`

- **ISA:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (not wgmma).
- **Tile:** 64×64×32, 128 thr / 4 warps (2×2), FP32 accum → BF16 store.
- **Split-K** when `ceil(M/64)*ceil(N/64)` ≪ 512 CTAs.
- **Fallback:** v4 for router + m≥192 + any TC failure.
- **Compile:** v8 `cpp_sources` forward-decl pattern; `prepare` prints OK/FAILED + warmup.

## When to spend a check (5 left today)

Only after a mental checklist:

1. [ ] Paper: every non-router scored shape m≤128 hits `gemm_nt_tc*` in profile (not only nvjet).
2. [ ] Paper: m=16/32/128 should not be 10–60× slower than SGLang (v8 failure mode).
3. [ ] If first check is correct but ~0.5–0.9×: tune tile/SK/double-buffer before another check.
4. [ ] If incorrect: fix fragment layout vs PTX table before any speed work.
5. [ ] Do **not** scored-submit until check ≥ **~1.10×** (clear of v4 CI).

## Likely next steps after a green check

- Double-buffer smem + cp.async (still mma.sync).
- Smaller BM for m≤16 (less zero-row waste).
- Optional wgmma path only if mma.sync plateaus below ~1.2×.
- Router fused hi+lo as a separate bet (big SOL gap, orthogonal to TC GEMM).
