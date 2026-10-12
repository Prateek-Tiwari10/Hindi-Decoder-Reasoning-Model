"""Part 5 -- intrinsic evaluation: cross-entropy, perplexity and bits-per-byte.

WHY REPORT BOTH PERPLEXITY AND BITS-PER-BYTE
--------------------------------------------
Perplexity is ``exp(mean cross-entropy per token)``, so its unit is "per token"
-- and a token is whatever the tokenizer says it is.  Every model in this study
shares the same 16K Hindi BPE tokenizer, so perplexity IS comparable across the
GPT- and LLaMA-style models here.  Bits-per-byte is reported alongside because
it is the tokenizer-independent unit: it stays valid if a model with a different
vocabulary is ever added, and it matches the units of published byte-level
numbers.

WHY BITS-PER-BYTE FIXES IT
--------------------------
BPB re-denominates the same total uncertainty in a unit both models share: the
raw UTF-8 bytes of the underlying text.

    BPB = (total cross-entropy in nats) / (ln 2 x number of UTF-8 bytes)

The numerator is the model's total surprise over the test set.  The denominator
is a property of the *text*, identical no matter how it was tokenized.  A
tokenizer that packs more text per token gets a proportionally larger numerator
per token, and the ratio cancels it out.  So BPB asks "how many bits does this
model need per byte of Devanagari?", which is answerable for both models on the
same scale.

The ``ln 2`` converts nats to bits.  Cross-entropy from ``F.cross_entropy`` is in
nats (natural log); "bits"-per-byte requires log base 2.

THE DENOMINATOR MUST BE BYTES, NOT TOKENS
------------------------------------------
The single easiest way to get BPB wrong is to divide by the token count, which
silently reproduces perplexity's problem in different clothes.  This module
decodes the scored tokens back to text and measures ``len(text.encode("utf-8"))``
-- see :func:`utf8_bytes_of`.

WHY EVALUATION WALKS THE STREAM SEQUENTIALLY
--------------------------------------------
Training samples random windows, which is right for training and wrong here: a
perplexity over random windows would score some tokens twice and skip others
entirely.  This module uses non-overlapping windows of exactly ``context_len``,
so consecutive targets are contiguous and every token in the split is scored
exactly once -- except token 0, which nothing precedes and which therefore
cannot be predicted.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import ModelConfig, Paths
from data.token_stream import TokenStream
from model import build_model

LOGGER = logging.getLogger("evaluate")

#: Tokens decoded per call when counting UTF-8 bytes.  Large enough that
#: SentencePiece's handling of the leading word-boundary marker at a chunk edge
#: is a rounding error, small enough not to build a 100 MB string.
DECODE_CHUNK = 1_000_000


def load_trained(language: str, checkpoint: Path, device: str,
                 root: Path = Path("."), variant: str = "16k"
                 ) -> tuple[torch.nn.Module, ModelConfig]:
    """Load a trained model from a checkpoint.

    Args:
        language: ``"hindi"`` (the only language).
        checkpoint: Path to the ``.pt`` file.
        device: Torch device.
        root: Repository root.
        variant: Architecture variant naming the config to check against.

    Returns:
        ``(model in eval mode, its ModelConfig)``.

    Raises:
        ValueError: If the checkpoint's architecture differs from the config on
            disk, which would mean evaluating a different model than the one
            the report describes.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(state["model_config"])

    arch = state.get("arch", "gpt")

    # The variant's config file describes the GPT-style variants; checkpoints
    # of another architecture carry their own config, which is used as is.
    if arch == "gpt":
        config_path = Paths.for_variant(language, root, variant).model_config
        on_disk = ModelConfig.load(config_path)
        if config.to_dict() != on_disk.to_dict():
            raise ValueError(
                f"{checkpoint} was trained with a different architecture than "
                f"{config_path} describes")

    model = build_model(config, device=device, arch=arch)
    model.load_state_dict(state["model"])
    model.eval()
    LOGGER.info("%s: loaded step %d from %s", language, state["step"], checkpoint)
    return model, config


def utf8_bytes_of(token_ids: np.ndarray, tokenizer) -> int:
    """UTF-8 byte count of the text a token sequence decodes to.

    This is the BPB denominator, and it must measure the *text*, not the
    tokens.  Decoding in chunks keeps peak memory bounded; the only cost is
    that SentencePiece drops a leading word-boundary space at each chunk
    boundary, which across a 34M-token split is on the order of 30 bytes in
    ~140 million.

    Args:
        token_ids: The token ids that were scored.
        tokenizer: A SentencePieceProcessor.

    Returns:
        Number of UTF-8 bytes.
    """
    total = 0
    for start in range(0, len(token_ids), DECODE_CHUNK):
        chunk = token_ids[start:start + DECODE_CHUNK].tolist()
        total += len(tokenizer.decode(chunk).encode("utf-8"))
    return total


@torch.no_grad()
def score_split(model: torch.nn.Module, stream: TokenStream, device: str,
                batch_size: int = 16, max_batches: int | None = None
                ) -> tuple[float, int, np.ndarray]:
    """Sum cross-entropy over every token of a split, each scored once.

    Args:
        model: Trained model in eval mode.
        stream: The split's token stream.
        device: Torch device.
        batch_size: Sequences per forward pass.
        max_batches: Stop early, for smoke tests.

    Returns:
        ``(total_ce_nats, tokens_scored, scored_token_ids)``.
    """
    total_ce = 0.0
    n_tokens = 0
    scored: list[np.ndarray] = []

    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) \
        if device.startswith("cuda") else torch.autocast("cpu", enabled=False)

    for i, (x, y) in enumerate(stream.iter_blocks(batch_size)):
        if max_batches is not None and i >= max_batches:
            break
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)

        with autocast:
            logits, _, _ = model(xb)

        # reduction="sum" in float32: averaging per batch then averaging the
        # averages would weight a short final batch equally with a full one.
        # The total is what BPB needs anyway.
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               yb.reshape(-1), reduction="sum")
        total_ce += loss.item()
        n_tokens += yb.numel()
        scored.append(y.reshape(-1))

    return total_ce, n_tokens, np.concatenate(scored)


def evaluate(language: str, split: str = "test", checkpoint: Path | None = None,
             device: str = "cpu", batch_size: int = 16,
             max_batches: int | None = None, root: Path = Path("."),
             variant: str = "16k") -> dict:
    """Compute CE, perplexity and BPB for one model on one split.

    Args:
        language: ``"hindi"`` (the only language).
        split: Which split to score; ``"test"`` for the reported numbers.
        checkpoint: Trained checkpoint; defaults to the variant's final model.
        device: Torch device.
        batch_size: Sequences per forward pass.
        max_batches: Stop early, for smoke tests.
        root: Repository root.
        variant: Architecture variant, selecting the config, checkpoint,
            tokenizer and bins.

    Returns:
        The measured metrics.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, variant)
    model, config = load_trained(
        language, checkpoint or paths.final_model, device, root, variant)
    stream = TokenStream(paths.split_bin(split), config.context_len)
    tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))

    total_ce, n_tokens, scored_ids = score_split(
        model, stream, device, batch_size, max_batches)

    n_bytes = utf8_bytes_of(scored_ids, tokenizer)
    mean_ce = total_ce / n_tokens

    return {
        "language": language,
        "split": split,
        "checkpoint": str(checkpoint or paths.final_model),
        "tokens_in_split": len(stream),
        "tokens_scored": n_tokens,
        "utf8_bytes": n_bytes,
        "bytes_per_token": n_bytes / n_tokens,
        "total_ce_nats": total_ce,
        "mean_ce_nats": mean_ce,
        "perplexity": math.exp(mean_ce),
        "bits_per_token": mean_ce / math.log(2),
        # The headline cross-model number: total nats over raw bytes of text.
        "bits_per_byte": total_ce / (math.log(2) * n_bytes),
    }


def comparison_table(results: list[dict]) -> str:
    """Render the side-by-side table the report needs.

    Args:
        results: One result per model.

    Returns:
        A markdown table.
    """
    head = ("| Model | Split | Tokens | UTF-8 bytes | CE (nats/tok) | "
            "PPL | BPB |\n|---|---|---:|---:|---:|---:|---:|\n")
    rows = "".join(
        f"| {r['language'].capitalize()} | {r['split']} | {r['tokens_scored']:,} | "
        f"{r['utf8_bytes']:,} | {r['mean_ce_nats']:.4f} | "
        f"{r['perplexity']:.2f} | {r['bits_per_byte']:.4f} |\n"
        for r in results)
    return head + rows


def main() -> None:
    """Evaluate a checkpoint and write the LM-metrics report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--split", default="test",
                        choices=("train", "val", "test"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Override the checkpoint; defaults to the "
                             "variant's final.pt")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/eval.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    languages = ("hindi",)

    results = []
    for language in languages:
        LOGGER.info("scoring %s %s ...", language, args.split)
        result = evaluate(language, args.split, checkpoint=args.checkpoint,
                          device=device, batch_size=args.batch_size,
                          max_batches=args.max_batches, root=args.root,
                          variant=args.variant)
        results.append(result)
        LOGGER.info("  CE %.4f nats/tok | PPL %.2f | BPB %.4f | %s bytes",
                    result["mean_ce_nats"], result["perplexity"],
                    result["bits_per_byte"], f"{result['utf8_bytes']:,}")

    table = comparison_table(results)
    LOGGER.info("\n%s", table)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"results": results, "table_markdown": table}, indent=2),
        encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
