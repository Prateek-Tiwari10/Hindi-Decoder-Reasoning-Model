"""Audit the cleaned corpus: manual token share, split sizes, source mix.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

THE QUESTION THIS ANSWERS
-------------------------
The spec requires **at least 20% of final training tokens to come from manual
collection**.  Every estimate of that share before this point has been a
projection; this module measures it on the actual cleaned corpus.

It is measured in **tokens, not documents**, and the distinction is not
pedantic.  Manual crawl documents average ~1,900 characters while Sangraha
documents average ~1,930, but the two corpora differ enough in length
distribution that a document share and a token share can disagree by several
points -- and it is the token share the spec grades.

WHY IT RUNS BEFORE THE TOKENIZER
--------------------------------
If the share is short, the fix is to keep all manual documents and subsample
public ones, which changes the corpus the tokenizer is trained on.  Training the
tokenizer first and auditing afterwards would mean throwing the tokenizer away.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from data.pipeline.dedup import all_clean_shards, iter_shard

LOGGER = logging.getLogger("audit")

#: Measured 2026-08-14, reports/data/chars_per_token_measurement.json.
CHARS_PER_TOKEN = {"hindi": 4.247}

#: The spec's floor.  The instructor has indicated 80-90M surviving tokens is
#: acceptable against a nominal 100M, but 20% remains the number to clear.
MANUAL_SHARE_FLOOR = 0.20

#: Below this the share is treated as too close to the floor to trust, since
#: chars/token is measured rather than exact and the final tokenizer may shift
#: it by a percent or two.
MANUAL_SHARE_COMFORT = 0.22


def audit_corpus(clean_dir: Path, chars_per_token: float) -> dict:
    """Measure token counts by split, source type and source.

    Args:
        clean_dir: Directory of cleaned ``train``/``val``/``test`` shards.
        chars_per_token: Measured constant for this language.

    Returns:
        Statistics including the manual token share.
    """
    chars_by_split: Counter = Counter()
    docs_by_split: Counter = Counter()
    chars_by_type: Counter = Counter()
    docs_by_type: Counter = Counter()
    chars_by_source: Counter = Counter()
    # Manual share is graded on the training corpus, which is what the model
    # actually learns from; val/test are reported separately.
    train_chars_by_type: Counter = Counter()

    for shard in all_clean_shards(clean_dir):
        split = shard.name.split("-")[0]
        for record in iter_shard(shard):
            text = record.get("text") or ""
            if not text:
                continue
            source_type = record.get("source_type") or "public"
            source = record.get("source") or "unknown"
            chars_by_split[split] += len(text)
            docs_by_split[split] += 1
            chars_by_type[source_type] += len(text)
            docs_by_type[source_type] += 1
            chars_by_source[source] += len(text)
            if split == "train":
                train_chars_by_type[source_type] += len(text)

    def tokens(chars: float) -> int:
        return int(chars / chars_per_token)

    train_manual = train_chars_by_type.get("manual", 0)
    train_public = train_chars_by_type.get("public", 0)
    train_total = train_manual + train_public
    manual_share = train_manual / train_total if train_total else 0.0

    return {
        "chars_per_token_measured": chars_per_token,
        "splits": {
            split: {"documents": docs_by_split[split],
                    "characters": chars_by_split[split],
                    "tokens": tokens(chars_by_split[split])}
            for split in sorted(chars_by_split)
        },
        "by_source_type": {
            source_type: {"documents": docs_by_type[source_type],
                          "characters": chars_by_type[source_type],
                          "tokens": tokens(chars_by_type[source_type])}
            for source_type in sorted(chars_by_type)
        },
        "by_source": {
            source: {"characters": chars_by_source[source],
                     "tokens": tokens(chars_by_source[source])}
            for source in sorted(chars_by_source, key=chars_by_source.get, reverse=True)
        },
        "train_manual_tokens": tokens(train_manual),
        "train_public_tokens": tokens(train_public),
        "train_total_tokens": tokens(train_total),
        "manual_share": round(manual_share, 4),
        "meets_floor": manual_share >= MANUAL_SHARE_FLOOR,
        "comfortable": manual_share >= MANUAL_SHARE_COMFORT,
    }


def subsample_plan(result: dict, target_tokens: int = 500_000_000) -> dict:
    """Compute how to assemble a target-sized corpus that clears the floor.

    All manual tokens are kept and public is subsampled to fill the remainder,
    which is what makes a large public pool an asset rather than a dilution: the
    manual share of the *assembled* corpus is chosen, not inherited.

    Args:
        result: Output of :func:`audit_corpus`.
        target_tokens: Desired final training-token count.

    Returns:
        The assembly plan and the manual share it achieves.
    """
    manual = result["train_manual_tokens"]
    public = result["train_public_tokens"]
    available = manual + public

    if available <= target_tokens:
        return {"feasible": available >= target_tokens * 0.95,
                "note": "corpus smaller than target; use all of it",
                "manual_tokens": manual, "public_tokens": public,
                "total_tokens": available,
                "manual_share": round(manual / available, 4) if available else 0.0}

    public_needed = max(0, target_tokens - manual)
    public_keep = min(public, public_needed)
    total = manual + public_keep
    return {
        "feasible": True,
        "target_tokens": target_tokens,
        "manual_tokens": manual,
        "public_tokens_kept": public_keep,
        "public_tokens_discarded": public - public_keep,
        "public_keep_fraction": round(public_keep / public, 4) if public else 0.0,
        "total_tokens": total,
        "manual_share": round(manual / total, 4) if total else 0.0,
    }


def main() -> None:
    """Audit one language's cleaned corpus."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True, choices=sorted(CHARS_PER_TOKEN))
    parser.add_argument("--clean", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    clean_dir = args.clean or Path(args.language) / "data" / "clean"
    result = audit_corpus(clean_dir, CHARS_PER_TOKEN[args.language])
    result["language"] = args.language
    result["assembly_plan"] = subsample_plan(result, args.target_tokens)

    report = args.report or Path(f"reports/data/audit_{args.language}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    share = result["manual_share"]
    LOGGER.info("=" * 64)
    LOGGER.info("%s: train %s tokens = %s manual + %s public",
                args.language, f"{result['train_total_tokens']:,}",
                f"{result['train_manual_tokens']:,}", f"{result['train_public_tokens']:,}")
    LOGGER.info("%s: MANUAL SHARE = %.2f%%  (floor 20%%)  %s",
                args.language, 100 * share,
                "PASS" if result["meets_floor"] else "FAIL")
    plan = result["assembly_plan"]
    if plan.get("feasible"):
        LOGGER.info("%s: assembled 500M corpus -> %.2f%% manual (keep %.1f%% of public)",
                    args.language, 100 * plan["manual_share"],
                    100 * plan.get("public_keep_fraction", 1.0))
    LOGGER.info("report -> %s", report)


if __name__ == "__main__":
    main()
