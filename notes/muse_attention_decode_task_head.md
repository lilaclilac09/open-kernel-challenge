# Gated attention decode (`muse_attention_decode`)

One decode step or 8-token verify block of grouped-query attention over a paged FP8 cache: QK-norm, a 2,048-token sliding window with RoPE or full attention over the whole context, and a gated, normalized output.

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_attention_decode`: it tells `okc submit` the problem.
- `baseline.py`, SGLang's kernel: the one you're timed against.
- `reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet. The judge uses its own copies, and runs them with steps of its own (the weights, routing fields, fresh inputs), so to check a change, use `okc check`.
- `AGENTS.md`, notes for a coding agent, if you use one.

## What to compute

```python
# per request b and block token j, at position p = positions[t]
q, k, v, gate = split(qkv[t])   # the fused projection output
q = rmsnorm(q) * 5.473;  k = rmsnorm(k)
q, k = rope(q, p), rope(k, p)   # sliding layers
cache[cache_loc[t]] = fp8(k), fp8(v)
a = softmax(q @ K.T / 8) @ V    # up to p: window or all
out = rmsnorm(a) * sigmoid(gate)
```

`kernel_fn(data)` is the timed call and returns bfloat16 [B, L, Hq·64]. The `Inputs` and `Config` classes in submission.py give every field's dtype and shape; return GPU tensors of exactly that shape and dtype.

## Shapes

- B is batch and L is q_len: 1 for a decode step, or 8 for a speculative verify block (causal inside it). Each GPU has 16 query heads and 2 key/value heads (TP8).
- Each load times a sliding-window layer (the last 2,048 tokens, with RoPE) and a full-attention layer (the whole context, no RoPE). Contexts are around 80K tokens.
- The inputs are what SGLang’s attention receives: the fused projection output, plus the step’s positions, cache rows and FA3 lengths, which SGLang builds once per step.
- `scheduler_metadata` is FA3's tile schedule for the step, built from its lengths as SGLang does: set on full-attention decode steps, None on sliding layers and verify blocks. Kernels that don't use FA3 can ignore it.
- The cache is FP8 E4M3. Its page size is 1 in the scored cases and 64 in most tests, so read config["page_size"]. Its scales are 1 in the scored cases and 1/64 in some tests.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

| load | shape | weight | floor |
|---|---|---|---|
| decode 1 | batch 1, ctx 80000, heads tp8, layer sliding, page 1, dist soft | 46 | 0.1 µs |
| decode 1 | batch 1, ctx 80000, heads tp8, layer full, page 1, dist coherent | 16 | 4.3 µs |
| decode 16 | batch 16, ctx 80000, heads tp8, layer sliding, page 1, spread agentic, dist soft | 46 | 1.6 µs |
| decode 16 | batch 16, ctx 80000, heads tp8, layer full, page 1, spread agentic, dist coherent | 16 | 63.2 µs |
| decode 32 | batch 32, ctx 80000, heads tp8, layer sliding, page 1, spread agentic, dist soft | 46 | 3.3 µs |
| decode 32 | batch 32, ctx 80000, heads tp8, layer full, page 1, spread agentic, dist coherent | 16 | 126.4 µs |
| verify 1 | batch 1, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, dist soft | 46 | 0.1 µs |
| verify 1 | batch 1, ctx 80000, heads tp8, layer full, page 1, q_len 8, dist coherent | 16 | 4.3 µs |
| verify 16 | batch 16, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 1.9 µs |
| verify 16 | batch 16, ctx 80000, heads tp8, layer full, page 1, q_len 8, spread agentic, dist coherent | 16 | 63.4 µs |
| verify 32 | batch 32, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 3.7 µs |
| verify 32 | batch 32, ctx 80000, heads tp8, layer full, page 1, q_len 8, spread agentic, dist coherent | 16 | 126.8 µs |

A shape's weight is how many times one step runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per step, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 16 tests, some harder than the scored shapes:

```
batch 1, ctx 10, heads tp8, layer sliding
batch 3, ctx 2047, heads tp8, layer sliding
batch 5, ctx 5000, heads tp8, layer sliding, q_len 8, spread mixed, page 1
batch 4, ctx 3000, heads tp8, layer full
batch 8, ctx 8192, heads dp, layer full, spread mixed
batch 2, ctx 100000, heads tp8, layer full, q_len 8, page 1
batch 4, ctx 6000, heads tp8, layer sliding, dist edges, page 1
batch 4, ctx 6000, heads tp8, layer full, q_len 8, dist edges
batch 1, ctx 1048576, heads tp8, layer full, q_len 8
batch 3, ctx 4000, heads tp8, layer sliding, q_len 8, page 1, kv_scale 0.015625
batch 2, ctx 9000, heads tp8, layer full, kv_scale 0.015625
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist soft
batch 4, ctx 6000, heads tp8, layer sliding, q_len 8, dist soft, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist soft
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist coherent, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist coherent
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The output: within 2.5× the reference's own error when its queries are rounded to FP8 E4M3, plus 1.5%, kept between 3% and 15%, against an FP32 reference.
- The new cache entries: within 8% of the reference's. Your output is checked against the reference attending to your own new entries.
