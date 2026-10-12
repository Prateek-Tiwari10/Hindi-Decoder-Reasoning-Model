"""Download the public (non-manual) portion of each corpus.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

The spec allows up to 80% of training tokens to come from existing public
corpora, with at least 20% manually collected.  This module fetches that public
portion and tags every document ``source_type="public"``, so the manual share can
be audited exactly (see :mod:`data.pipeline.audit`).

Sources, and why these:

* **Sangraha ``verified``** (AI4Bharat) -- the largest cleaned Indic corpus.
  Only the ``verified`` split is used.  The ``synthetic`` split is
  machine-translated from English, and training a *monolingual* model on
  translationese would corrupt the language model.
* **Wikipedia** -- smaller but encyclopedic register, which balances the heavy
  news skew of the crawl.  Also the training data for the language identifier.

Shards are **read over HTTP range requests, never downloaded**.  Sangraha's Hindi
split is 37 GB across 100 shards,
while this machine has ~4 GB of free RAM.  Reading parquet row groups on demand
means the transfer stops the moment the token budget is met, so a few hundred MB
moves instead of tens of GB and nothing large ever touches disk.

The cost of that choice is a slow start: a large shard's parquet footer takes
tens of seconds to read before any rows arrive.  That is paid once per shard and
is irrelevant against a 400M-token budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from data.pipeline.crawl import ShardWriter
from data.pipeline.normalize import devanagari_ratio

LOGGER = logging.getLogger("public_corpus")

#: Average characters per token for a 32k Devanagari BPE, used only to convert
#: the token budget into a character budget while downloading.
#:
#: This 4.0 was a guess.  Measured on 2026-08-14 with a throwaway 32k BPE (see
#: ``reports/data/chars_per_token_measurement.json``): **Hindi 4.166** aggregated
#: over the corpus's real source mix.  Pass ``--chars-per-token`` with the
#: measured value; this constant remains only as the default for callers that
#: have not measured yet.
CHARS_PER_TOKEN_ESTIMATE = 4.0


@dataclass
class PublicSource:
    """One downloadable public corpus.

    Attributes:
        name: Short identifier recorded as provenance on every document.
        repo_id: HuggingFace dataset repository.
        path_prefix: Repository path prefix selecting this language's shards.
        text_column: Parquet column holding the document text.
        weight: Share of the language's public budget to draw from this source.
            Wikipedia is capped well below Sangraha because it is much smaller
            and would otherwise be exhausted and over-represented.
    """

    name: str
    repo_id: str
    path_prefix: str
    text_column: str = "text"
    weight: float = 1.0


#: Per-language public source registry. Language codes differ per repository
#: (Sangraha uses ``hin``/``nep``; Wikipedia uses dated ``20231101.hi`` configs),
#: which is why these are spelled out rather than derived.
PUBLIC_SOURCES: dict[str, list[PublicSource]] = {
    "hindi": [
        PublicSource("sangraha_verified", "ai4bharat/sangraha", "verified/hin/", weight=0.85),
        PublicSource("wikipedia", "wikimedia/wikipedia", "20231101.hi/", weight=0.15),
    ],
}


def list_shards(source: PublicSource) -> list[str]:
    """List the parquet shards belonging to a source.

    Args:
        source: Source whose shards to enumerate.

    Returns:
        Sorted repository paths of the source's parquet files.
    """
    files = HfApi().list_repo_files(source.repo_id, repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith(source.path_prefix) and f.endswith(".parquet")
    )


def stream_shard(
    fs: HfFileSystem, remote_path: str, text_column: str, batch_size: int = 2000
) -> Iterator[list[str]]:
    """Yield document texts from a remote parquet shard, in batches.

    The shard is read **over HTTP range requests** rather than downloaded.  This
    is not an optimization but a necessity: a single Sangraha shard can be
    several GB, and the budget is satisfied long before that shard is
    exhausted.  Reading row groups on demand means the transfer
    stops when the budget is met, so a few hundred MB moves instead of tens of
    GB, and nothing large ever touches disk.

    Args:
        fs: HuggingFace filesystem handle.
        remote_path: Path of the form ``datasets/<repo>/<file>``.
        text_column: Column holding document text.
        batch_size: Rows read per batch, keeping peak memory bounded.

    Yields:
        Lists of document strings.
    """
    # cache_type="readahead" is essential, not cosmetic. fsspec's default cache
    # retains every byte range it has fetched, so a long sequential scan of a
    # multi-GB shard grows without bound -- observed at 2.4 GB RSS and heavy
    # swapping on this 14 GB machine. Readahead keeps only a sliding window.
    handle = fs.open(remote_path, "rb", block_size=8 * 1024 * 1024,
                     cache_type="readahead")
    parquet = pq.ParquetFile(handle)
    if text_column not in parquet.schema_arrow.names:
        # Sangraha and Wikipedia both use "text", but fall back to the first
        # string column rather than failing the whole download.
        candidates = [
            n for n, t in zip(parquet.schema_arrow.names, parquet.schema_arrow.types)
            if "string" in str(t)
        ]
        if not candidates:
            raise ValueError(f"no string column in {remote_path}")
        LOGGER.warning("column %r absent; using %r", text_column, candidates[0])
        text_column = candidates[0]

    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_column]):
            yield [t for t in batch.column(0).to_pylist() if t]
    finally:
        handle.close()


def download_source(
    language: str,
    source: PublicSource,
    out_dir: Path,
    char_budget: int,
    min_chars: int = 200,
    min_devanagari_ratio: float = 0.6,
    skip_shards: int = 0,
    chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE,
) -> dict:
    """Download one public source until its character budget is met.

    Shards are fetched, streamed to compressed JSONL, and then deleted from the
    HuggingFace cache before the next shard is fetched, so peak disk stays at
    roughly one shard rather than the whole dataset.

    Args:
        language: ``"hindi"``, recorded as provenance.
        source: Source to download.
        out_dir: Destination directory for output shards.
        char_budget: Stop once this many characters have been written.
        min_chars: Skip documents shorter than this.
        min_devanagari_ratio: Skip documents below this Devanagari character
            ratio, which removes English and romanized rows that survive in
            even curated Indic corpora.
        skip_shards: Ignore this many leading shards.  Shards are consumed in a
            fixed sorted order, so a top-up run that skips the shards an earlier
            run already consumed reads only text this corpus has never seen --
            without which a second run would re-download the same rows from
            shard 0 and hand the deduplicator the work instead.

            Note the order is *lexicographic*, not numeric: ``data-0``,
            ``data-1``, ``data-10``, ``data-11``, ... ``data-2``.  So
            ``skip_shards=2`` skips ``data-0`` and ``data-1`` and resumes at
            ``data-10``, which looks wrong and is not -- ``data-2`` through
            ``data-9`` are unread and simply come later in this ordering.  What
            matters for correctness is that the ordering is *stable* across runs,
            so "skip the first N" always names the same N shards.

    Returns:
        Statistics for this source.
    """
    shards = list_shards(source)
    if not shards:
        LOGGER.error("%s/%s: no shards found at %r",
                     language, source.name, source.path_prefix)
        return {"source": source.name, "documents": 0, "characters": 0, "shards_used": 0}

    n_available = len(shards)
    if skip_shards:
        if skip_shards >= n_available:
            LOGGER.error("%s/%s: skip-shards=%d leaves nothing of %d shards",
                         language, source.name, skip_shards, n_available)
            return {"source": source.name, "documents": 0, "characters": 0,
                    "shards_used": 0, "shards_skipped": skip_shards,
                    "shards_available": n_available}
        LOGGER.info("%s/%s: skipping %d already-consumed shard(s)",
                    language, source.name, skip_shards)
        shards = shards[skip_shards:]

    LOGGER.info("%s/%s: %d shards available (%d after skip), budget %.0fM chars",
                language, source.name, n_available, len(shards), char_budget / 1e6)

    writer = ShardWriter(out_dir, prefix=f"public-{source.name}", max_bytes=256 * 1024 * 1024)
    n_docs = 0
    n_chars = 0
    n_skipped = 0
    shards_used = 0
    started = time.time()

    fs = HfFileSystem()

    try:
        for shard_path in shards:
            if n_chars >= char_budget:
                break
            remote = f"datasets/{source.repo_id}/{shard_path}"
            shards_used += 1
            try:
                for texts in stream_shard(fs, remote, source.text_column):
                    for text in texts:
                        if len(text) < min_chars:
                            n_skipped += 1
                            continue
                        if devanagari_ratio(text) < min_devanagari_ratio:
                            n_skipped += 1
                            continue
                        writer.write({
                            "url": None,
                            "host": None,
                            "source": source.name,
                            "source_type": "public",
                            "language": language,
                            "fetched_at": time.time(),
                            "title": "",
                            "text": text,
                        })
                        n_docs += 1
                        n_chars += len(text)
                    if n_chars >= char_budget:
                        break
            except Exception as exc:
                LOGGER.warning("%s: read failed: %s", shard_path, exc)
            finally:
                # Arrow buffers the row groups it has read; release them before
                # opening the next shard so peak RSS stays flat across shards.
                gc.collect()

            elapsed = max(time.time() - started, 1e-6)
            LOGGER.info(
                "  %s shard %d/%d | docs=%s chars=%.1fM (%.0f%% of budget) %.1fMB/s",
                source.name, shards_used, len(shards), f"{n_docs:,}",
                n_chars / 1e6, 100 * n_chars / char_budget,
                n_chars / 1e6 / elapsed,
            )
    finally:
        writer.close()

    return {
        "source": source.name,
        "documents": n_docs,
        "characters": n_chars,
        "estimated_tokens": int(n_chars / chars_per_token),
        "documents_skipped": n_skipped,
        "shards_used": shards_used,
        "shards_skipped": skip_shards,
        "shards_available": n_available,
    }


def run(
    language: str,
    out_dir: Path,
    target_tokens: int,
    report_path: Path,
    only_sources: list[str] | None = None,
    skip_shards: int = 0,
    chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE,
) -> None:
    """Download every public source for a language against a token budget.

    Args:
        language: ``"hindi"``.
        out_dir: Destination directory for output shards.
        target_tokens: Total public-token budget for this language.
        report_path: Where to write the download statistics JSON.
        only_sources: If given, download only these source names.  A top-up run
            uses this to draw the shortfall from a source that still has shards
            left, rather than from one already exhausted.
        skip_shards: Leading shards to skip in each selected source; see
            :func:`download_source`.
        chars_per_token: Conversion used to turn the token budget into a
            character budget.  Pass the measured value for this language --
            leaving the 4.0 default slightly over-estimates the budget.
    """
    sources = PUBLIC_SOURCES[language]
    if only_sources:
        known = {s.name for s in sources}
        unknown = sorted(set(only_sources) - known)
        if unknown:
            raise SystemExit(
                f"unknown source(s) for {language}: {', '.join(unknown)} "
                f"(available: {', '.join(sorted(known))})"
            )
        sources = [s for s in sources if s.name in only_sources]

    total_chars = target_tokens * chars_per_token
    # Weights are shares of the full public budget. When only a subset of
    # sources runs, renormalize so the requested budget is actually drawn
    # rather than silently scaled down by the omitted sources' weights.
    weight_total = sum(s.weight for s in sources) or 1.0
    LOGGER.info("%s: public budget %.0fM tokens (~%.1f GB of text) from %s",
                language, target_tokens / 1e6, total_chars / 1e9,
                ", ".join(s.name for s in sources))

    results = []
    for source in sources:
        budget = int(total_chars * source.weight / weight_total)
        results.append(download_source(language, source, out_dir, budget,
                                       skip_shards=skip_shards,
                                       chars_per_token=chars_per_token))
        LOGGER.info("%s/%s done: %s", language, source.name, results[-1])

    summary = {
        "language": language,
        "target_tokens": target_tokens,
        "chars_per_token_used": chars_per_token,
        "sources": results,
        "total_documents": sum(r["documents"] for r in results),
        "total_characters": sum(r["characters"] for r in results),
        "total_estimated_tokens": sum(r.get("estimated_tokens", 0) for r in results),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    LOGGER.info("=" * 62)
    LOGGER.info("%s: %s docs, %.1fM chars, ~%.1fM tokens estimated",
                language, f"{summary['total_documents']:,}",
                summary["total_characters"] / 1e6,
                summary["total_estimated_tokens"] / 1e6)
    LOGGER.info("report -> %s", report_path)


def main() -> None:
    """Parse arguments and download a language's public corpus."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True, choices=sorted(PUBLIC_SOURCES))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=400_000_000,
                        help="public-token budget (default: 80%% of 500M)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--source", action="append", dest="sources", default=None,
                        help="restrict to this source (repeatable); default: all")
    parser.add_argument("--skip-shards", type=int, default=0,
                        help="skip this many leading shards per source, so a "
                             "top-up run does not re-read consumed shards")
    parser.add_argument("--chars-per-token", type=float,
                        default=CHARS_PER_TOKEN_ESTIMATE,
                        help="measured chars/token for this language "
                             "(Hindi 4.166 as of 2026-08-14)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    report = args.report or Path(f"reports/data/public_download_{args.language}.json")
    run(args.language, args.out, args.target_tokens, report,
        only_sources=args.sources, skip_shards=args.skip_shards,
        chars_per_token=args.chars_per_token)


if __name__ == "__main__":
    main()
