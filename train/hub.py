"""Package a trained checkpoint for the Hugging Face Hub.

WHAT GETS PUBLISHED AND WHAT DOES NOT
--------------------------------------
A training checkpoint is 310-334 MB because it carries the optimiser's two
moment estimates per parameter plus the GradScaler and RNG state -- everything
needed to *resume*.  None of that is needed to *use* the model, and publishing
it triples the download for no benefit.  This strips the checkpoint to weights
alone (~104 MB in fp32) and writes the config beside them, so the published
artifact is self-describing.

The resumable checkpoints stay where they are; they are the training record.

WHY safetensors
---------------
``torch.load`` on an untrusted file executes arbitrary pickle opcodes.  Anything
published for other people to download should not require them to trust it, so
weights go out as safetensors, which is a flat tensor container with no code
path.  ``config.json`` carries the architecture so the weights can be rebuilt
with ``model.gpt.build_model`` and nothing else.

WEIGHT TYING AND THE SHARED TENSOR
-----------------------------------
The output head *is* the token embedding transposed, so the two names point at
one tensor.  safetensors refuses to serialise aliased storage -- silently
writing it twice would double the file and, worse, let the two copies drift on
reload.  The head is therefore dropped from the payload and re-tied by
``build_model`` at load time, which is where the tying belongs anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from model.config import ModelConfig, Paths

LOGGER = logging.getLogger("hub")


def strip_to_weights(checkpoint: Path) -> tuple[dict, ModelConfig, dict]:
    """Read a training checkpoint and return just what inference needs.

    Args:
        checkpoint: A ``.pt`` written by :meth:`train.pretrain.Trainer.save_checkpoint`.

    Returns:
        ``(weights, model config, provenance)`` where provenance records the
        step and validation loss the weights came from.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(state["model_config"])

    weights = {k: v.contiguous() for k, v in state["model"].items()}

    # Drop the tied head: it aliases the embedding, and build_model re-ties it.
    tied = [k for k in weights if k.startswith("lm_head")]
    for key in tied:
        weights.pop(key)

    provenance = {
        "step": state.get("step"),
        "val_loss": state.get("val_loss"),
        "train_config": state.get("train_config"),
        "tied_head_dropped": tied,
    }
    return weights, config, provenance


def package(language: str, variant: str, out_dir: Path,
            root: Path = Path("."), checkpoint: Path | None = None) -> Path:
    """Write a publishable model directory for one language and variant.

    Args:
        language: ``"hindi"`` (the only language).
        variant: ``""`` for the 24K baseline, ``"16k"`` for the narrow-deep model.
        out_dir: Directory to fill.
        root: Repository root.
        checkpoint: Override the checkpoint; defaults to the variant's final.

    Returns:
        The directory written.
    """
    from safetensors.torch import save_file

    paths = Paths.for_variant(language, root, variant)
    source = checkpoint or paths.final_model
    weights, config, provenance = strip_to_weights(source)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out_dir / "model.safetensors"))

    (out_dir / "config.json").write_text(
        json.dumps({**config.to_dict(),
                    "model_type": "decoder_only_transformer",
                    "architecture": "custom (lm/model.py); not a transformers class"},
                   indent=2) + "\n", encoding="utf-8")
    (out_dir / "training.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    tokenizer = paths.tokenizer
    (out_dir / tokenizer.name).write_bytes(tokenizer.read_bytes())
    vocab = tokenizer.with_suffix(".vocab")
    if vocab.exists():
        (out_dir / vocab.name).write_bytes(vocab.read_bytes())

    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    LOGGER.info("%s/%s -> %s (%.0f MB, from step %s)",
                language, variant or "24k", out_dir, size, provenance["step"])
    return out_dir


def verify(out_dir: Path, root: Path = Path(".")) -> dict:
    """Rebuild the model from the published files and check it works.

    Publishing weights that cannot be loaded back is a silent failure -- the
    upload succeeds and only a user discovers the problem.  This reconstructs
    the model from ``config.json`` plus ``model.safetensors`` alone, with no
    reference to the original checkpoint, and runs a forward pass.

    Args:
        out_dir: A directory written by :func:`package`.
        root: Repository root.

    Returns:
        What was checked.

    Raises:
        AssertionError: If the rebuilt model does not match its config.
    """
    from safetensors.torch import load_file

    from model import build_model

    config = ModelConfig.from_dict(json.loads((out_dir / "config.json").read_text()))
    weights = load_file(str(out_dir / "model.safetensors"))

    model = build_model(config, device="cpu", verify_budget=False)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # The tied head is the only thing allowed to be missing: build_model ties it.
    assert not unexpected, f"unexpected tensors in the published weights: {unexpected}"
    assert all(m.startswith("lm_head") for m in missing), \
        f"weights missing beyond the tied head: {missing}"

    model.eval()
    with torch.no_grad():
        logits, _, _ = model(torch.randint(0, config.vocab_size, (2, 16)))
    assert tuple(logits.shape) == (2, 16, config.vocab_size), logits.shape

    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "expected_parameters": config.parameter_count,
        "logits_shape": list(logits.shape),
        "re_tied_at_load": missing,
        "loads_from_published_files_alone": True,
    }


def main() -> None:
    """Package one or more models for the Hub."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory to write <variant>/<language>/ under")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    languages = ("hindi",)
    tag = args.variant or "24k"
    for language in languages:
        directory = package(language, args.variant, args.out / tag / language,
                            args.root)
        result = verify(directory, args.root)
        LOGGER.info("  verified: %s params (expected %s), logits %s, re-tied %s",
                    f"{result['parameters']:,}",
                    f"{result['expected_parameters']:,}",
                    result["logits_shape"], result["re_tied_at_load"])


if __name__ == "__main__":
    main()
