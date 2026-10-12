"""Publish the no-positional-embedding ablation to the Hugging Face Hub.

WHY THIS RUNS INSIDE THE TRAINING KERNEL
----------------------------------------
The trained checkpoint lives in the Kaggle session's output.  Downloading it,
then re-uploading from a laptop, moves ~100 MB twice over a link that has
already failed once in this project.  Pushing from inside the session moves it
once, straight from the machine that produced it, and means the evaluation
kernel can pull the model by name instead of depending on kernel outputs.

THE TOKEN
---------
Read from a Kaggle Secret named ``HF_TOKEN`` (Add-ons -> Secrets), falling back
to the environment.  It is never printed, never written to a file, and never
committed.  Use a token scoped to *write* on this one repository and revoke it
after the run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Public so the evaluation kernel -- and the grader -- can load it by name.
#: Published Hub repo -- its name predates the bonus->ablation rename and is
#: kept so existing links resolve.
DEFAULT_REPO = "Prateek-Tiwari10/LMA_bonus_nopos_hindi"


def read_token() -> str:
    """Return the Hugging Face write token, or raise with a usable message."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:                                   # not on Kaggle, or unset
        pass
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "No Hugging Face token.  On Kaggle: Add-ons -> Secrets -> add "
            "HF_TOKEN (write scope) and attach it to this notebook.")
    return token


def model_card(summary: dict, repo: str) -> str:
    """Compose the model card, quoting only numbers the run actually produced."""
    val = summary.get("final_val_loss")
    best = summary.get("best_val_loss")
    steps = summary.get("final_step")
    return f"""---
language: hi
license: mit
tags:
  - ablation
  - positional-embeddings
  - hindi
---

# LMA ablation — Hindi without positional embeddings

An 11-layer decoder-only Transformer trained on Hindi with the learned
absolute positional table **removed entirely** (`positional: "none"`), so the
model has no explicit representation of token order.

Everything else matches the standard Hindi model it is compared against — the
same 16K tokenizer, the same corpus and split, the same one-epoch horizon, and
the same optimiser recipe (peak LR 3e-3, 750 warmup steps, 32 x 2 x 512 tokens
per step). Training-set size, schedule and seed are held constant, so the only
variable is the positional table.

| | standard | ablated (this model) |
| --- | ---: | ---: |
| parameters | 25,860,480 | 25,663,872 |
| positional table | 512 x 384 learned | none |
| layers / d_model / heads | 11 / 384 / 6 | 11 / 384 / 6 |
| steps | 16,125 (1 epoch) | {steps if steps else 'n/a'} |
| final val loss | 3.4033 | {f'{val:.4f}' if isinstance(val, float) else 'n/a'} |
| best val loss | 3.4033 | {f'{best:.4f}' if isinstance(best, float) else 'n/a'} |

## What removing the table does and does not do

It does **not** make the model order-blind. Causal attention is itself a
positional signal: with the mask in place, position *t* attends to 0..*t*, so
representations at different positions are built from different prefixes. A
one-layer no-positional model is genuinely permutation-invariant over its
prefix, but from two layers onward that invariance is gone. This model
therefore has to *infer* order from the causal mask rather than read it from an
embedding — a weaker and less direct signal, which is what the evaluation
measures.

## Loading

```python
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import json

d = snapshot_download("{repo}")
cfg = json.load(open(f"{{d}}/config.json"))
weights = load_file(f"{{d}}/model.safetensors")
```

The architecture is `model/gpt.py` in the project repository, not a
`transformers` class. `build_model` re-ties the output head to the token
embedding, which is why no `lm_head` tensor is stored.
"""


def push(model_dir: Path, summary: dict, repo: str = DEFAULT_REPO) -> str:
    """Upload a packaged model directory, returning the repository URL."""
    from huggingface_hub import HfApi, create_repo

    token = read_token()
    (model_dir / "README.md").write_text(model_card(summary, repo),
                                         encoding="utf-8")
    (model_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    create_repo(repo, token=token, private=False, exist_ok=True,
                repo_type="model")
    HfApi().upload_folder(folder_path=str(model_dir), repo_id=repo,
                          token=token, repo_type="model",
                          commit_message="no-positional-embedding ablation, "
                                         "Hindi, one epoch")
    url = f"https://huggingface.co/{repo}"
    print(f"pushed {model_dir.name} -> {url}", flush=True)
    return url
