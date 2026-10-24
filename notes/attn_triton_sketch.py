# User pasted a Triton attention sketch (saved as notes).
# Prefer official baseline + FlashInfer/FA3 for first check:
# muse_attention_decode/submission.py currently = baseline.py + okc header.
# Sketch issues to fix later: cache indexing *0, missing page_table, window/full,
# FA3 scheduler_metadata. Do NOT use .item() (CUDA graph).
