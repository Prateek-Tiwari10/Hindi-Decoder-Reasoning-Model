"""Materialise a Hugging Face model as a checkpoint the Phase 2 tools can read.

WHY A CONVERSION IS NEEDED
--------------------------
What goes to the Hub is inference-only: safetensors weights plus a config, with
the tied output head dropped.  What ``eval.lm_metrics`` and friends read is a
training checkpoint -- a pickle carrying ``model``, ``model_config`` and
``step``.  Rather than teach every evaluation script a second input format,
this rebuilds the checkpoint shape from the published files.

The head is not restored here: ``build_model`` re-ties it at load time, which
is where the tying belongs.
"""

from __future__ import annotations

import json
from pathlib import Path


def fetch(repo: str, into: Path) -> Path:
    """Download a model repository, returning the local snapshot directory."""
    from huggingface_hub import snapshot_download

    into.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo_id=repo, local_dir=str(into),
                              allow_patterns=["*.safetensors", "*.json",
                                              "*.model", "*.vocab"])
    print(f"fetched {repo} -> {local}", flush=True)
    return Path(local)


def to_checkpoint(snapshot: Path, checkpoint: Path,
                  expect_positional: str | None = None) -> dict:
    """Write a ``.pt`` the Phase 2 loaders accept, and return its config dict.

    Args:
        snapshot: Directory from :func:`fetch`.
        checkpoint: Path to write.
        expect_positional: If given, refuse to proceed unless the published
            config says exactly this.  Guards against evaluating the standard
            model while believing it is the ablation, which would otherwise
            produce a perfectly plausible and completely wrong comparison.

    Returns:
        The model config as published.
    """
    import torch
    from safetensors.torch import load_file

    config = json.loads((snapshot / "config.json").read_text())
    if expect_positional is not None and config.get("positional") != expect_positional:
        raise SystemExit(
            f"{snapshot} publishes positional={config.get('positional')!r} but "
            f"{expect_positional!r} was expected; wrong repository")

    weights = load_file(str(snapshot / "model.safetensors"))

    # RE-TIE THE OUTPUT HEAD.  Publishing drops it because safetensors refuses
    # aliased storage, and ``build_model`` re-ties it when constructing from a
    # config.  But ``eval.lm_metrics.load_trained`` calls ``load_state_dict``
    # strictly against an already-built model, which expects the key to be
    # there.  A pickle has no objection to two names for one tensor, so the
    # alias is restored here rather than relaxing the strict load -- strictness
    # is what catches a genuinely mismatched checkpoint.
    if config.get("tie_weights", True) and "lm_head.weight" not in weights:
        embedding = weights.get("token_embedding.weight")
        if embedding is None:
            raise SystemExit(
                f"{snapshot} has neither lm_head.weight nor "
                "token_embedding.weight; the weights are unusable")
        weights["lm_head.weight"] = embedding

    provenance = {}
    training = snapshot / "training.json"
    if training.exists():
        provenance = json.loads(training.read_text())

    # Strip the derived extras ModelConfig.to_dict appends; they are outputs,
    # and feeding them back to the constructor raises.
    payload = {k: v for k, v in config.items()
               if k not in ("d_k", "parameter_count", "model_type",
                            "architecture")}

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": weights,
                "model_config": payload,
                "step": provenance.get("step", 0),
                "val_loss": provenance.get("val_loss")},
               checkpoint)
    print(f"  -> {checkpoint} (positional={payload.get('positional')!r}, "
          f"step={provenance.get('step', 'unknown')})", flush=True)
    return payload
