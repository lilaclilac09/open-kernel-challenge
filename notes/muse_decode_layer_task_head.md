# Whole decoder layer (`muse_decode_layer`)

Everything one GPU does for one layer of Muse Spark Open under TP8/EP8: the projections, attention, the router and experts, and every norm and residual in between.

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_decode_layer`: it tells `okc submit` the problem.
- `baseline.py`, SGLang's kernel: the one you're timed against.
- `reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet. The judge uses its own copies, and runs them with steps of its own (the weights, routing fields, fresh inputs), so to check a change, use `okc check`.
- `nvfp4.py`, the NVFP4 helper (the judge supplies its own copy).
- `AGENTS.md`, notes for a coding agent, if you use one.

## What to compute

```python
# x: attention input [T, 8192] BF16   s: residual stream FP32
q, k, v, g = split(x @ qkv_gate.T);  QK-norm, RoPE, write fp8(k), fp8(v)
a = (rmsnorm(attention(q)) * sigmoid(g)) @ o.T     # all-reduce: identity
s = α1 * s + β1 * rmsnorm(a);  h = rmsnorm(s) * (1 + w)
experts = top8(sigmoid(0.177 * h @ router.T) + bias)
y = local experts(rmsnorm(h @ pooler.T) * (1 + w))  # all-reduce: identity
f = y @ unpooler.T   # T ≤ 8: [y @ unpooler[:1024].T, 0, …, 0]
s = α2 * s + β2 * rmsnorm(f * (1 + w))
return rmsnorm(s) * (1 + w), s, experts
```

`kernel_fn(data, state)` is the timed call and returns (bfloat16 [T, 8192], float32 [T, 8192], int32 [T, 8]). `state` is whatever `prepare(weights)` returned: it runs once, untimed, for up to 300 s. The `Inputs`, `Config` and `Weights` classes in submission.py give every field's dtype and shape; return GPU tensors of exactly that shape and dtype.

## Shapes

- T is the tokens per step, batch × q_len (B × L). Each load times a sliding-window and a full-attention layer, at contexts around 80K tokens.
- Communication is left out: each all-reduce and all-gather is the identity on this GPU’s partial result, as if the other seven GPUs contributed nothing.
- `scheduler_metadata` is FA3's tile schedule for the step, built from its lengths as SGLang does: set on full-attention decode steps, None on sliding layers and verify blocks. Kernels that don't use FA3 can ignore it.
- Routing is skewed, as in the real model; a few experts per layer take most tokens.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

| load | shape | weight | floor |
|---|---|---|---|
| decode 1 | batch 1, ctx 80000, layer sliding, page 1, dist soft | 46 | 40.0 µs |
| decode 1 | batch 1, ctx 80000, layer full, page 1, dist soft | 16 | 44.2 µs |
| decode 16 | batch 16, ctx 80000, layer sliding, page 1, spread agentic, dist soft | 46 | 76.7 µs |
| decode 16 | batch 16, ctx 80000, layer full, page 1, spread agentic, dist soft | 16 | 137.5 µs |
| decode 32 | batch 32, ctx 80000, layer sliding, page 1, spread agentic, dist soft | 46 | 84.5 µs |
| decode 32 | batch 32, ctx 80000, layer full, page 1, spread agentic, dist soft | 16 | 208.1 µs |
| verify 1 | batch 1, ctx 80000, layer sliding, page 1, q_len 8, dist soft | 46 | 62.3 µs |
| verify 1 | batch 1, ctx 80000, layer full, page 1, q_len 8, dist soft | 16 | 66.4 µs |
| verify 16 | batch 16, ctx 80000, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 126.6 µs |
| verify 16 | batch 16, ctx 80000, layer full, page 1, q_len 8, spread agentic, dist soft | 16 | 188.7 µs |
| verify 32 | batch 32, ctx 80000, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 143.8 µs |
| verify 32 | batch 32, ctx 80000, layer full, page 1, q_len 8, spread agentic, dist soft | 16 | 267.2 µs |

A shape's weight is how many times one step runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per step, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 13 tests, some harder than the scored shapes:

```
batch 1, ctx 10, layer sliding
batch 1, ctx 3000, layer full
batch 8, ctx 2047, layer sliding
batch 9, ctx 5000, layer sliding, spread mixed, page 1
batch 3, ctx 6000, layer full, q_len 8, dist outliers
batch 5, ctx 5000, layer sliding, q_len 8, spread mixed, page 1
batch 33, ctx 20000, layer full, spread mixed, page 1
batch 2, ctx 300000, layer sliding, q_len 8
batch 1, ctx 1048576, layer full, q_len 8
batch 4, ctx 7000, layer sliding, q_len 8, page 1, kv_scale 0.015625
batch 1, ctx 80000, layer full, dist soft, page 1
batch 4, ctx 6000, layer sliding, q_len 8, dist soft
batch 16, ctx 20000, layer full, spread mixed, dist soft, page 1
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The residual update (output stream − α1·α2 · input stream) and the next attention input, against an FP32 reference: within 2.5× the reference's own error when its queries are rounded to FP8, which comes to 3–15% depending on the shape.
- The new cache entries: within 8% of the reference's.
- Any single row: at most 5× its bound. A near-zero row is measured against a tenth of the average row.
