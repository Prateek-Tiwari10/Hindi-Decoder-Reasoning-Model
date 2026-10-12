# Tests and correctness gates

| File | What it checks | Cost |
|---|---|---|
| `test_model_checks.py` | parameter count (analytic = live), output shapes, context cap, **future-blindness**, causal-mask shape, head split round-trip, initial loss ≈ ln V | CPU, seconds |
| `test_pretrain_gates.py` | **overfit one batch** (Gate A), **kill-and-resume** reproduces the uninterrupted loss trajectory (Gate B) | CPU/GPU, minutes |
| `test_finetune_gates.py` | prompt mask, no double shift, overfit, **kill-and-resume**, EOS stop, tokenizer round-trip, padding invariance (P1–P7) | GPU recommended |
| `verify_gpt_baselines.py` | the fixed GPT baselines reproduce their recorded numbers from this layout: test loss 3.4080 and held-out-name EM 0.954 | GPU, ~30 min |

Run as modules from the repository root, e.g. `python -m tests.test_model_checks`.
