"""Independent fertility measurement for the final tokenizers.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

WHY THIS EXISTS SEPARATELY FROM THE TRAINING REPORT
---------------------------------------------------
``data.pipeline.tokenizer`` measures fertility as a side effect of training, inside the
same process that produced the model.  That number is the one quoted in the
report, and it is not wrong -- but "the trainer graded its own homework" is a
fair question to be asked in a viva.  This module re-measures from scratch: it
loads the saved ``.model`` file from disk, re-reads the held-out text from the
cleaned shards, and recomputes every statistic.  Agreement between the two is
evidence the reported numbers are reproducible rather than incidental.

It also measures the four *sweep* models of each language against the same text,
so the final model can be shown to beat the candidates it was chosen over, on
identical held-out data.

WHICH SPLIT
-----------
**val only.**  The tokenizer trained on train, so measuring there would flatter
it.  The test split is never opened -- not in training and not here -- so it
stays untouched for Phase 2 reporting.  Reading it "just to evaluate the
tokenizer" is exactly how a held-out set stops being held out.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sentencepiece as spm

from data.pipeline.tokenizer import evaluate_tokenizer, export_split_text

LOGGER = logging.getLogger("eval_tokenizer")

#: Split used for every measurement here.  Not configurable on purpose: see the
#: module docstring.  Test is not an option.
EVAL_SPLIT = "val"


def load_eval_lines(clean_dir: Path, work_dir: Path, language: str) -> list[str]:
    """Export the complete val split and return it as lines.

    Unlike the training-time evaluation, which caps at a fixed line count, this
    uses the whole split.  The val splits are ~7-8M characters, so "all of it"
    is both affordable and the least arbitrary choice.

    Args:
        clean_dir: Directory of cleaned shards.
        work_dir: Scratch directory for the exported text.
        language: For the filename.

    Returns:
        Non-empty held-out lines.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{language}_{EVAL_SPLIT}_eval.txt"
    if not path.exists():
        export_split_text(clean_dir, EVAL_SPLIT, path)
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def measure(model_path: Path, lines: list[str]) -> dict:
    """Measure one model, adding the vocabulary size actually loaded.

    Args:
        model_path: A SentencePiece ``.model`` file.
        lines: Held-out lines.

    Returns:
        Measured statistics.
    """
    stats = evaluate_tokenizer(model_path, lines)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    stats["vocab_size_loaded"] = processor.get_piece_size()
    stats["model_path"] = str(model_path)
    # Bytes per token is the number that actually predicts storage cost, and it
    # is not recoverable from chars/token alone: Devanagari averages ~2.6 UTF-8
    # bytes per character, unlike the ~1.0 that Latin-script intuition assumes.
    n_bytes = sum(len(line.encode("utf-8")) for line in lines)
    stats["bytes_per_token"] = n_bytes / stats["tokens"] if stats["tokens"] else 0.0
    return stats


def run(language: str, clean_dir: Path, final_report: Path, sweep_dir: Path | None,
        work_dir: Path) -> dict:
    """Re-measure a language's final tokenizer, and its sweep candidates.

    Args:
        language: ``"hindi"``.
        clean_dir: Cleaned corpus directory.
        final_report: The final training report, naming the chosen model.
        sweep_dir: Directory of sweep models, if available.
        work_dir: Scratch for the exported held-out text.

    Returns:
        Measurements for the final model and every sweep model found.
    """
    lines = load_eval_lines(clean_dir, work_dir, language)
    LOGGER.info("%s: %s held-out %s lines", language, f"{len(lines):,}", EVAL_SPLIT)

    report = json.loads(final_report.read_text())
    chosen = report["chosen"]
    final = measure(Path(chosen["model_path"]), lines)
    LOGGER.info("%s final %s/%d: fertility=%.4f chars/token=%.4f unk=%.1e",
                language, chosen["model_type"], chosen["vocab_size"],
                final["tokens_per_word"], final["chars_per_token"],
                final["unk_rate"])

    # The number the trainer reported, for comparison. These should agree to
    # within rounding; a real gap would mean the model on disk is not the model
    # that was measured, which is worth knowing loudly.
    final["reported_tokens_per_word"] = chosen["tokens_per_word"]
    final["reported_chars_per_token"] = chosen["chars_per_token"]
    final["matches_training_report"] = (
        abs(final["tokens_per_word"] - chosen["tokens_per_word"]) < 5e-3)

    candidates = []
    for path in sorted((sweep_dir or Path("/nonexistent")).glob("*.model")):
        stats = measure(path, lines)
        stats["name"] = path.stem
        candidates.append(stats)
        LOGGER.info("  sweep %-28s fertility=%.4f", path.stem,
                    stats["tokens_per_word"])

    return {
        "language": language,
        "eval_split": EVAL_SPLIT,
        "eval_lines": len(lines),
        "note": ("Measured by re-loading the saved model and re-reading the "
                 "held-out val split. The test split is never read."),
        "final": final,
        "chosen": {"model_type": chosen["model_type"],
                   "vocab_size": chosen["vocab_size"],
                   "trained_on": report.get("train_chars_used")},
        "sweep_candidates": candidates,
    }


def main() -> None:
    """Re-measure both languages and write one combined report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--languages", nargs="+", default=["hindi"])
    parser.add_argument("--work", type=Path, default=Path("/tmp/lma_eval"))
    parser.add_argument("--report", type=Path,
                        default=Path(f"reports/data/tokenizer_evaluation.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    results = {}
    for language in args.languages:
        results[language] = run(
            language=language,
            clean_dir=Path(language) / "data" / "clean",
            final_report=Path(f"reports/data/tokenizer_final_{language}.json"),
            sweep_dir=Path(f"sweep_outputs/{language}_tokenizer_sweep"),
            work_dir=args.work,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", args.report)

    for language, data in results.items():
        final = data["final"]
        print(f"\n{language.upper()}  ({data['eval_lines']:,} held-out "
              f"{data['eval_split']} lines, {final['characters']:,} chars)")
        print(f"  final {data['chosen']['model_type']} "
              f"{data['chosen']['vocab_size']:,}  "
              f"(trained on {data['chosen']['trained_on']})")
        print(f"    fertility      {final['tokens_per_word']:.4f} tokens/word")
        print(f"    chars/token    {final['chars_per_token']:.4f}")
        print(f"    bytes/token    {final['bytes_per_token']:.4f}")
        print(f"    UNK rate       {final['unk_rate']:.1e}")
        print(f"    reproduces the training report: "
              f"{final['matches_training_report']} "
              f"(reported {final['reported_tokens_per_word']:.4f})")
        if data["sweep_candidates"]:
            print(f"  {'sweep candidate':28s} {'fertility':>10s} {'vs final':>10s}")
            for c in sorted(data["sweep_candidates"],
                            key=lambda c: c["tokens_per_word"]):
                delta = (c["tokens_per_word"] - final["tokens_per_word"]) \
                    / final["tokens_per_word"] * 100
                print(f"    {c['name']:26s} {c['tokens_per_word']:10.4f} "
                      f"{delta:+9.2f}%")


if __name__ == "__main__":
    main()
