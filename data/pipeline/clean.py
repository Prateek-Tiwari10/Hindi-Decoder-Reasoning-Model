"""Cleaning pipeline orchestrator: raw shards in, tokenizer-ready splits out.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

STAGE ORDER -- this is graded and is deliberate
-----------------------------------------------
    normalize -> hold out test/val -> LID -> language cleaning
              -> quality -> dedup -> decontaminate

The one that looks wrong and is not: **test and validation are held out BEFORE
cleaning, not after deduplication.**  Splitting after cleaning would mean the
evaluation sets have been through exactly the same filters as train, so they
measure generalization to *surviving* text rather than to real text.  Holding out
first costs one thing -- near-duplicates could straddle the split -- and that is
repaid with interest by the decontamination pass, which removes any train
document resembling a held-out one, including paraphrases that a split-after-
dedup design would leave sitting in train.

WHY TWO PHASES
--------------
Phase A is per-document and embarrassingly parallel, so it runs across all cores
and streams to staged shards.  Phase B (dedup) is inherently global -- it must
see every document before it can know what is a duplicate -- so it runs once,
single-process, over the staged output.  Splitting them this way is what keeps
peak memory bounded on a machine with a few GB free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

from data.pipeline.dedup import ShardWriterSimple, deduplicate_corpus, iter_shard
from data.pipeline.lid import LanguageIdentifier, LidModel
from data.pipeline.normalize import DevanagariNormalizer, NormalizerConfig, devanagari_ratio
from data.pipeline.quality import QualityConfig, QualityFilter

LOGGER = logging.getLogger("clean")

#: Measured 2026-08-14, reports/data/chars_per_token_measurement.json. Used only for
#: token accounting in the statistics; never for a filtering decision.
CHARS_PER_TOKEN = {"hindi": 4.166}

#: LID label per language.
#: The label the Hindi-vs-Nepali language-ID filter must assign for a
#: document to be KEPT in the Hindi corpus.
LID_LABEL = {"hindi": "hi"}

#: Target size of each held-out split, in tokens.
HELDOUT_TOKENS = 5_000_000

#: Sentence-level Latin-script removal (stage 4). A sentence is dropped only if
#: it is BOTH clearly non-Devanagari AND long enough to judge; short or ambiguous
#: fragments are kept, because a quoted name or an acronym is not noise.
SENTENCE_MIN_WORDS = 8
SENTENCE_MIN_DEVANAGARI = 0.30


@dataclass
class StageStats:
    """Per-stage document and character counts for the statistics report."""

    documents_in: int = 0
    characters_in: int = 0
    dropped_lid: int = 0
    dropped_lid_ambiguous: int = 0
    dropped_quality: int = 0
    quality_reasons: dict[str, int] = field(default_factory=dict)
    sentences_seen: int = 0
    sentences_removed: int = 0
    kept: int = 0
    characters_kept: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "StageStats") -> None:
        """Accumulate another worker's counters into this one."""
        self.documents_in += other.documents_in
        self.characters_in += other.characters_in
        self.dropped_lid += other.dropped_lid
        self.dropped_lid_ambiguous += other.dropped_lid_ambiguous
        self.dropped_quality += other.dropped_quality
        self.sentences_seen += other.sentences_seen
        self.sentences_removed += other.sentences_removed
        self.kept += other.kept
        self.characters_kept += other.characters_kept
        for key, value in other.quality_reasons.items():
            self.quality_reasons[key] = self.quality_reasons.get(key, 0) + value
        for key, value in other.split_counts.items():
            self.split_counts[key] = self.split_counts.get(key, 0) + value


def assign_split(text: str, val_share: float, test_share: float) -> str:
    """Assign a document to train/val/test by deterministic hash.

    Hash-based rather than random so the assignment is reproducible and
    stateless: the same document always lands in the same split, no matter how
    many workers run or in what order shards are read.

    Args:
        text: Normalized document text.
        val_share: Fraction of documents for validation.
        test_share: Fraction for test.

    Returns:
        ``"train"``, ``"val"`` or ``"test"``.
    """
    # First 8 hex digits of SHA-1 give a uniform value in [0, 1).
    bucket = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < val_share:
        return "val"
    if bucket < val_share + test_share:
        return "test"
    return "train"


def clean_sentences(text: str, stats: StageStats) -> str:
    """Remove confidently non-Devanagari sentences from a document (stage 4).

    Runs only on documents that already passed document-level LID, and is
    deliberately conservative -- see ``SENTENCE_MIN_WORDS``.  Hindi-vs-Nepali is
    never decided at this level; this only separates Devanagari from Latin.

    Args:
        text: Normalized document text.
        stats: Counters to update, so the removal fraction can be reported.

    Returns:
        The document with offending sentences removed.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        for sentence in line.split("।"):
            sentence = sentence.strip()
            if not sentence:
                continue
            stats.sentences_seen += 1
            words = sentence.split()
            if (len(words) >= SENTENCE_MIN_WORDS
                    and devanagari_ratio(sentence) < SENTENCE_MIN_DEVANAGARI):
                stats.sentences_removed += 1
                continue
            kept.append(sentence)
    return "। ".join(kept)


_WORKER: dict = {}


def _init_worker(language: str, lid_path: str) -> None:
    """Build the per-process normalizer, LID model and quality filter.

    Loading the 6.8 MB LID model once per worker rather than once per document is
    the difference between minutes and hours on a 2M-document corpus.
    """
    config = NormalizerConfig.for_hindi()
    _WORKER["normalizer"] = DevanagariNormalizer(config)
    _WORKER["lid"] = LanguageIdentifier(LidModel.load(lid_path))
    _WORKER["label"] = LID_LABEL[language]
    # Hindi uses the default (strict) quality thresholds: Hindi web text is
    # abundant, so a borderline document is cheap to discard.
    _WORKER["quality"] = QualityFilter(QualityConfig())
    _WORKER["language"] = language


def _process_shard(args: tuple[str, str, float, float]) -> tuple[str, StageStats]:
    """Run stages 1-5 over one shard and write the survivors to a staged shard.

    Args:
        args: ``(shard_path, out_dir, val_share, test_share)``.

    Returns:
        ``(output_prefix, stats)`` for the caller to merge.
    """
    shard_path, out_dir, val_share, test_share = args
    stats = StageStats()
    normalizer = _WORKER["normalizer"]
    identifier = _WORKER["lid"]
    quality = _WORKER["quality"]
    want = _WORKER["label"]

    prefix = f"staged-{Path(shard_path).stem.replace('.jsonl', '')}"
    writer = ShardWriterSimple(Path(out_dir), prefix)

    for record in iter_shard(Path(shard_path)):
        raw = record.get("text") or ""
        if not raw:
            continue
        stats.documents_in += 1
        stats.characters_in += len(raw)

        # Stage 1 -- normalize.
        text = normalizer.normalize(raw)

        # Stage 2 -- split assignment, before any filtering.
        split = assign_split(text, val_share, test_share)

        # Stage 3 -- document-level LID. Wrong language or ambiguous: drop.
        prediction = identifier.predict(text)
        if prediction.label != want:
            if prediction.label == "ambiguous":
                stats.dropped_lid_ambiguous += 1
            stats.dropped_lid += 1
            continue

        # Stage 4 -- sentence-level Latin-script removal.
        text = clean_sentences(text, stats)

        # Stage 5 -- quality filters. check() returns None on pass, or a short
        # named reason on rejection, which is what the per-reason removal table
        # in the statistics report is built from.
        reason = quality.check(text)
        if reason is not None:
            stats.dropped_quality += 1
            stats.quality_reasons[reason] = stats.quality_reasons.get(reason, 0) + 1
            continue

        record["text"] = text
        record["split"] = split
        record["lid_margin"] = round(prediction.ngram_logratio, 4)
        writer.write(record)
        stats.kept += 1
        stats.characters_kept += len(text)
        stats.split_counts[split] = stats.split_counts.get(split, 0) + 1

    writer.close()
    return prefix, stats


def run(language: str, data_dir: Path, out_dir: Path, work_dir: Path,
        report_path: Path, workers: int) -> dict:
    """Run the whole pipeline for one language.

    Args:
        language: ``"hindi"``.
        data_dir: Directory containing ``public/`` and ``manual/``.
        out_dir: Destination for cleaned train/val/test shards.
        work_dir: Scratch space.
        report_path: Where to write the statistics JSON.
        workers: Parallel worker processes for phase A.

    Returns:
        Combined statistics.
    """
    started = time.time()
    chars_per_token = CHARS_PER_TOKEN[language]
    shards = sorted((data_dir / "public").glob("*.jsonl.zst")) + \
        sorted((data_dir / "manual").glob("*.jsonl.zst"))
    LOGGER.info("%s: %d input shards", language, len(shards))

    # Held-out shares are computed from the target token counts, using the
    # measured chars/token, so val and test land near 5M tokens each rather than
    # at an arbitrary percentage.
    #
    # The two conversions below must not be confused, and were on the first run:
    # zstd's ~5.5x ratio is BYTES to BYTES, while the token estimate needs
    # CHARACTERS. Devanagari costs ~2.5 UTF-8 bytes per character, so multiplying
    # compressed bytes by 5.5 and calling the result "characters" over-counts by
    # that factor -- it produced a 3.07x token over-estimate, and held-out splits
    # came out at 1.7M tokens instead of 5M.
    ZSTD_RATIO = 5.5          # compressed bytes -> uncompressed bytes
    BYTES_PER_CHAR = 2.5      # uncompressed bytes -> characters (Devanagari)
    total_bytes = sum(p.stat().st_size for p in shards) * ZSTD_RATIO
    estimated_tokens = (total_bytes / BYTES_PER_CHAR) / chars_per_token
    share = min(0.02, HELDOUT_TOKENS / max(estimated_tokens, 1))
    LOGGER.info("%s: ~%.0fM tokens estimated, held-out share %.4f each",
                language, estimated_tokens / 1e6, share)

    staged_dir = work_dir / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    lid_path = str(Path("data/pipeline/models/lid_hi_ne.json"))

    stats = StageStats()
    tasks = [(str(p), str(staged_dir), share, share) for p in shards]
    with mp.Pool(workers, initializer=_init_worker, initargs=(language, lid_path)) as pool:
        for done, (_prefix, shard_stats) in enumerate(
                pool.imap_unordered(_process_shard, tasks), start=1):
            stats.merge(shard_stats)
            LOGGER.info("  phase A %d/%d shards | kept %s docs",
                        done, len(tasks), f"{stats.kept:,}")

    LOGGER.info("phase A done in %.0fs: %s -> %s docs",
                time.time() - started, f"{stats.documents_in:,}", f"{stats.kept:,}")

    staged = sorted(staged_dir.glob("*.jsonl.zst"))
    train_shards, heldout_shards = _partition_staged(staged)
    dedup_stats = deduplicate_corpus(
        train_shards=train_shards, heldout_shards=heldout_shards,
        out_dir=out_dir, work_dir=work_dir / "dedup",
        chars_per_token=chars_per_token)

    summary = {
        "language": language,
        "cleaned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chars_per_token_measured": chars_per_token,
        "input_shards": len(shards),
        "stage_a": {
            "documents_in": stats.documents_in,
            "characters_in": stats.characters_in,
            "tokens_in": int(stats.characters_in / chars_per_token),
            "dropped_lid_total": stats.dropped_lid,
            "dropped_lid_ambiguous": stats.dropped_lid_ambiguous,
            "dropped_quality": stats.dropped_quality,
            "quality_rejections_by_reason": stats.quality_reasons,
            "sentences_seen": stats.sentences_seen,
            "sentences_removed": stats.sentences_removed,
            "sentence_removal_fraction": (stats.sentences_removed /
                                          max(stats.sentences_seen, 1)),
            "documents_kept": stats.kept,
            "characters_kept": stats.characters_kept,
            "split_counts": stats.split_counts,
        },
        "stage_b_dedup": dedup_stats,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", report_path)
    return summary


def _partition_staged(staged: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split staged shards into train-bearing and held-out-bearing lists.

    Staged shards contain a mix of splits, so both lists are the same files; the
    dedup driver reads the ``split`` field per record.  Returning them as
    (train, []) keeps every document in one pass while the cluster resolver
    still distinguishes held-out documents by their ``split`` field.
    """
    return staged, []


def main() -> None:
    """Parse arguments and clean one language."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True, choices=sorted(CHARS_PER_TOKEN))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    data = args.data or Path(args.language) / "data"
    run(language=args.language,
        data_dir=data,
        out_dir=args.out or data / "clean",
        work_dir=args.work or data / "work",
        report_path=args.report or Path(f"reports/data/clean_{args.language}.json"),
        workers=args.workers)


if __name__ == "__main__":
    main()
