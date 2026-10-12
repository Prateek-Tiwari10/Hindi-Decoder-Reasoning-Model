"""Phase 3 — what finetuning cost the model on the data it was pretrained on.

THE QUESTION THIS ANSWERS
--------------------------
``eval.reasoning_perplexity`` scores every model on the *reasoning* corpus, which
measures how well finetuning fitted the new task.  It cannot say what that
fitting cost, because the pretrained base was never asked to model reasoning
text in the first place.

The complementary measurement is the reverse: run the finetuned checkpoints on
the **Phase 2 held-out news/web split** -- the distribution the base was
actually trained on -- and compare against the base.  A rise in perplexity there
is catastrophic forgetting, and it is invisible to every other metric in this
report.

WHY THE BASE IS RE-MEASURED RATHER THAN QUOTED
-----------------------------------------------
Phase 2 reports the base over the full 35.9M-token split.  Scoring five models
per language over the full split is hours on one laptop GPU, so a fixed prefix
is used instead -- and the base is re-scored on *that same prefix*.  Quoting
Phase 2's full-split figure beside a subset figure for the finetunes would be
comparing two different measurements and calling the difference an effect.
The full-split number is carried alongside only as a sanity check.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from model.config import Paths
from eval.lm_metrics import evaluate

LOGGER = logging.getLogger("eval.forgetting")

LANGUAGES = ("hindi",)

#: Sequences per forward pass; sized for a 4 GB card.
BATCH = 8
#: 500 x 8 x 512 = 2.05M tokens, the same prefix of the same split for every
#: model, which is what makes the rows comparable to each other.
MAX_BATCHES = 500

#: Phase 2's full-split figures, for reference only -- see the module docstring.
PHASE2_FULL_SPLIT = {
    "hindi": {"perplexity": 30.2033, "bits_per_byte": 0.4892},
}


def parse_model_args(items: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``TAG=PATH`` command-line items.

    Args:
        items: Raw ``--model`` values.

    Returns:
        ``(tag, path)`` pairs.
    """
    pairs = []
    for item in items:
        tag, _, path = item.partition("=")
        if not tag or not path:
            raise SystemExit(f"--model expects TAG=PATH, got {item!r}")
        pairs.append((tag, Path(path)))
    return pairs


def main() -> None:
    """Score base and finetunes on the Phase 2 test split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_finetune/forgetting.json"))
    parser.add_argument("--model", action="append", default=[],
                        metavar="TAG=PATH",
                        help="extra checkpoint to score (repeatable); the "
                             "GPT baseline pretrained + ft_60k are always scored")
    args = parser.parse_args()

    root = args.root.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload: dict = {}

    for language in LANGUAGES:
        paths = Paths.for_variant(language, root, "16k")
        checkpoints = {"pretrained": paths.final_model,
                       "ft_60k": paths.gpt_finetuned}
        checkpoints.update(dict(parse_model_args(args.model)))

        payload[language] = {"phase2_full_split_reference": PHASE2_FULL_SPLIT[language],
                             "protocol": {"split": "test",
                                          "batch_size": args.batch_size,
                                          "max_batches": args.max_batches},
                             "models": {}}
        base = None
        for tag, path in checkpoints.items():
            if not path.exists():
                LOGGER.warning("skipping %s: %s missing", tag, path)
                continue
            result = evaluate(language, split="test", checkpoint=path,
                              device=device, batch_size=args.batch_size,
                              max_batches=args.max_batches, root=root,
                              variant="16k")
            cell = {k: result[k] for k in
                    ("tokens_scored", "utf8_bytes", "mean_ce_nats",
                     "perplexity", "bits_per_token", "bits_per_byte")
                    if k in result}
            if tag == "pretrained":
                base = cell
            else:
                cell["perplexity_ratio_vs_base"] = (
                    cell["perplexity"] / base["perplexity"]) if base else None
                cell["bpb_delta_vs_base"] = (
                    cell["bits_per_byte"] - base["bits_per_byte"]) if base else None
            payload[language]["models"][tag] = cell
            LOGGER.info("%-7s %-10s ppl %9.2f  bpb %.4f%s", language[:2], tag,
                        cell["perplexity"], cell["bits_per_byte"],
                        f"   x{cell['perplexity_ratio_vs_base']:.2f} vs base"
                        if tag != "pretrained" and base else "")
            if device == "cuda":
                torch.cuda.empty_cache()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
