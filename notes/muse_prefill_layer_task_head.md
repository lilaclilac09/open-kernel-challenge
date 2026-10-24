# Prefill layer (`muse_prefill_layer`)

The whole decoder layer at the prefill shapes of an agentic coding turn: 753 new tokens on an 80K-token conversation, four conversations at once, and the first turn’s 16K-token chunks. Most of the time goes to the routed experts, as one large grouped NVFP4 GEMM.

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_prefill_layer`: it tells `okc submit` the problem.
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

- Each case is one prefill forward: B requests, each adding L new tokens (causal within the block) after seq_lens[b] cached ones, written into the paged FP8 cache.
- Inputs, outputs, bounds and the routing check are muse_decode_layer’s. The cache’s page size is 1, except one test with 64-token pages, so read config["page_size"].
- Prefill isn’t part of a decode step, so this problem has its own board and doesn’t count toward the overall board.

| load | shape | weight | floor |
|---|---|---|---|
| turn 1 | batch 1, ctx 80000, layer sliding, q_len 753, page 1, dist soft | 46 | 257.0 µs |
| turn 1 | batch 1, ctx 80000, layer full, q_len 753, page 1, dist soft | 16 | 496.7 µs |
| turn 4 | batch 4, ctx 80000, layer sliding, q_len 753, page 1, dist soft | 46 | 1,036.3 µs |
| turn 4 | batch 4, ctx 80000, layer full, q_len 753, page 1, dist soft | 16 | 1,992.4 µs |
| first turn | batch 1, ctx 0, layer full, q_len 16384, page 1, dist soft | 16 | 5,518.4 µs |
| first turn | batch 1, ctx 32768, layer sliding, q_len 16384, page 1, dist soft | 138 | 5,663.4 µs |
| first turn | batch 1, ctx 32768, layer full, q_len 16384, page 1, dist soft | 16 | 7,710.0 µs |
| first turn | batch 1, ctx 65536, layer full, q_len 16384, page 1, dist soft | 16 | 9,847.7 µs |

A shape's weight is how many times one prefill load runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per prefill load, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 5 tests, some harder than the scored shapes:

```
batch 1, ctx 0, layer sliding, q_len 300, page 1
batch 1, ctx 3000, layer full, q_len 753, page 1
batch 2, ctx 2500, layer sliding, q_len 753, page 1
batch 3, ctx 5000, layer full, q_len 200, dist outliers
batch 1, ctx 4096, layer sliding, q_len 4096, page 1
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The residual update (output stream − α1·α2 · input stream) and the next attention input, against an FP32 reference: within 2.5× the reference's own error when its queries are rounded to FP8, which comes to 3–15% depending on the shape.
- The new cache entries: within 8% of the reference's.
- Any single row: at most 5× its bound. A near-zero row is measured against a tenth of the average row.
- No NaN or Inf.

Routing may differ from the reference where scores nearly tie:

- Each expert you pick must score (sigmoid + router_bias) within 0.05 of the reference's 8th best.
- Each of the reference's top 8 on this GPU that beats its 9th best by more than 0.03 (0.04 above 256 tokens) must be among yours.
- This GPU must get at least the reference's routes minus 8% and 4, and its distinct experts minus 15% and 2.
- The MoE output is then checked with the experts you picked.

## How a run is timed

- Each shape is captured as a CUDA graph and timed by replaying it, in the same run and on the same GPU as SGLang's kernel, your replays and SGLang's taking turns on the same fresh inputs.
- Every timed replay starts with a cold L2: the judge reads a buffer larger than L2 just before it.
- A shape's speedup is the median of paired ratios: each repetition's SGLang time over yours, taken two repetitions at a time with the order swapped, so which side went first cancels. Your time on the shape is SGLang's median time over that speedup.
