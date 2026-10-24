# Muse Spark Open Kernel Challenge

Work for [openkernelchallenge.dev](https://openkernelchallenge.dev): make Meta’s Muse Spark Open faster by beating SGLang’s GPU kernels (then serving).

## Track

- **Kernel** (now → Nov 4, 11:59 pm PT): optimize five kernels on H200 vs SGLang baseline.
- **Serving** (Nov 5–19 PT): patch SGLang on 8×H200 for agentic coding load (1–32 concurrent users). Prizes go here.

## Problems

| Problem | What |
| --- | --- |
| `muse_bf16_proj` | Dense BF16 projections |
| `muse_attention_decode` | Gated attention decode over FP8 KV |
| `muse_moe_w4a16` | Routed MoE experts (NVFP4) |
| `muse_decode_layer` | Whole decoder layer |
| `muse_prefill_layer` | Prefill layer |

Each folder is from `okc init`: `submission.py` (start here), `baseline.py` (SGLang), `reference.py`, `TASK.md`, `AGENTS.md`.

## CLI

```bash
curl -fsSL https://openkernelchallenge.dev/install.sh | sh
export PATH="$HOME/.okc/bin:$PATH"
okc login
okc submit muse_moe_w4a16/submission.py
```

Submissions are Apache-2.0 (challenge terms).
