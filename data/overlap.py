"""Measure contamination between the train, validation and test splits.

WHY THIS HAS TO BE MEASURED, NOT ASSUMED
----------------------------------------
Phase 1 held the validation and test splits out *before* cleaning and then
decontaminated train against them.  That is the right order, but it is a claim
about a pipeline, and the numbers in Part 5 are only meaningful if the claim
actually holds on the bytes now on disk.  If a document sits in both train and
test, the test perplexity is partly memorisation and the whole evaluation is
soft.  So it is checked here directly.

TWO KINDS OF OVERLAP
--------------------
1. **Exact** -- the identical document text in two splits.  Caught with SHA-1
   over every document in all three splits.  This is definitive and cheap, and
   it is the failure mode that matters most: a whole document leaking across
   the boundary.

2. **Near-duplicate** -- the same story reworded, a syndicated article
   republished with a different headline, a page re-crawled after a small edit.
   Exact hashing misses all of these.  Caught with MinHash over word 5-gram
   shingles plus banded LSH -- the same algorithm, the same shingle size and the
   same hash family that ``common/dedup.py`` used to build the corpus, so this
   measures the pipeline on its own terms.

WHAT THIS DOES *NOT* CLAIM
--------------------------
Short spans shared between splits -- a common phrase, a stock news sentence, a
boilerplate footer -- are not contamination and are not counted.  Natural text
shares n-grams; that is a property of language, not a leak.  What would be a
leak is a whole document, or a document that is substantially the same
document, appearing on both sides, and that is what the two measures above
detect.

The ``.bin`` files inherit whatever the cleaned splits contain: each split's
``.bin`` is built only from that split's shards, so token overlap between
``train.bin`` and ``test.bin`` can only come from documents that overlap.
Measuring at the document level therefore settles the question for the token
streams too.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.pipeline.dedup import (BANDS, ROWS, MinHasher, iter_shard, jaccard,
                          sha1_of, shingles, split_shards)

LOGGER = logging.getLogger("overlap")

SPLITS = ("train", "val", "test")

#: Estimated Jaccard at or above which two documents are called near-duplicates.
#: 0.8 is the conventional line: below it documents share topic and phrasing,
#: above it one is a rewrite of the other.
NEAR_DUPLICATE_THRESHOLD = 0.8


def band_keys(signature: np.ndarray, bands: int = BANDS, rows: int = ROWS
              ) -> list[tuple]:
    """Split a MinHash signature into per-band hash keys.

    Two documents become LSH candidates if they agree on *any* whole band.  A
    single band of ``rows`` values agreeing by chance is unlikely unless the
    documents are genuinely similar, and using ``bands`` of them makes a miss
    unlikely for truly similar pairs.

    Args:
        signature: uint32 signature of length ``bands * rows``.
        bands: Number of bands.
        rows: Rows per band.

    Returns:
        One hashable key per band.
    """
    return [tuple(signature[b * rows:(b + 1) * rows].tolist())
            for b in range(bands)]


def index_heldout(clean_dir: Path, hasher: MinHasher) -> dict:
    """Read val and test, and build the structures train is checked against.

    Args:
        clean_dir: Cleaned corpus directory.
        hasher: Shared MinHash family.

    Returns:
        Mapping with the exact-hash table, the shingle sets, the signatures and
        the LSH band index.
    """
    exact: dict[str, tuple[str, int]] = {}
    shingle_sets: list[np.ndarray] = []
    refs: list[tuple[str, int]] = []
    buckets: dict[tuple, list[int]] = defaultdict(list)
    counts: dict[str, int] = {}

    for split in ("val", "test"):
        n = 0
        for shard in split_shards(clean_dir, split):
            for doc in iter_shard(shard):
                text = doc.get("text", "")
                if not text:
                    continue
                # A repeated document inside one split is a dedup question, not
                # a contamination question; keep the first occurrence.
                exact.setdefault(sha1_of(text), (split, n))

                sh = shingles(text)
                signature = hasher.signature(sh)
                idx = len(refs)
                refs.append((split, n))
                shingle_sets.append(sh)
                for key in band_keys(signature):
                    buckets[key].append(idx)
                n += 1
        counts[split] = n
        LOGGER.info("indexed %s: %s documents", split, f"{n:,}")

    return {"exact": exact, "shingles": shingle_sets, "refs": refs,
            "buckets": buckets, "counts": counts}


def scan_train(clean_dir: Path, index: dict, hasher: MinHasher,
               sample_every: int) -> dict:
    """Stream train and look for documents that leak into val or test.

    Args:
        clean_dir: Cleaned corpus directory.
        index: Structures from :func:`index_heldout`.
        hasher: Shared MinHash family.
        sample_every: Check every Nth document for near-duplicates.  Exact
            hashing is cheap and always runs on every document; MinHash is not,
            so the near-duplicate sweep is sampled.

    Returns:
        Counts and examples of both kinds of overlap.
    """
    exact_hits: list[dict] = []
    near_hits: list[dict] = []
    n_docs = n_sampled = 0

    for shard in split_shards(clean_dir, "train"):
        for doc in iter_shard(shard):
            text = doc.get("text", "")
            if not text:
                continue
            n_docs += 1

            hit = index["exact"].get(sha1_of(text))
            if hit is not None and len(exact_hits) < 20:
                exact_hits.append({"train_doc": n_docs - 1,
                                   "found_in": hit[0],
                                   "url": doc.get("url", ""),
                                   "preview": text[:120]})
            elif hit is not None:
                exact_hits.append({"train_doc": n_docs - 1, "found_in": hit[0]})

            if n_docs % sample_every:
                continue
            n_sampled += 1

            sh = shingles(text)
            if sh.size == 0:
                continue
            signature = hasher.signature(sh)

            # Union the candidates from every band, then score each candidate
            # once with the exact Jaccard of the shingle sets.  LSH proposes;
            # the exact score decides.
            candidates: set[int] = set()
            for key in band_keys(signature):
                candidates.update(index["buckets"].get(key, ()))

            for cand in candidates:
                score = jaccard(sh, index["shingles"][cand])
                if score >= NEAR_DUPLICATE_THRESHOLD:
                    split, held_idx = index["refs"][cand]
                    near_hits.append({"train_doc": n_docs - 1,
                                      "found_in": split,
                                      "heldout_doc": held_idx,
                                      "jaccard": round(float(score), 4),
                                      "preview": text[:120]})
                    break

    return {"train_documents": n_docs,
            "train_documents_minhashed": n_sampled,
            "exact_overlap_count": len(exact_hits),
            "exact_overlap_examples": exact_hits[:20],
            "near_duplicate_count": len(near_hits),
            "near_duplicate_examples": near_hits[:20]}


def val_test_overlap(clean_dir: Path) -> dict:
    """Check the two held-out splits against each other, exactly.

    Val and test must also be disjoint: if they share documents, tuning on val
    quietly tunes on test.

    Args:
        clean_dir: Cleaned corpus directory.

    Returns:
        The count of shared documents.
    """
    seen: dict[str, str] = {}
    shared = 0
    for split in ("val", "test"):
        for shard in split_shards(clean_dir, split):
            for doc in iter_shard(shard):
                text = doc.get("text", "")
                if not text:
                    continue
                digest = sha1_of(text)
                if seen.get(digest, split) != split:
                    shared += 1
                seen.setdefault(digest, split)
    return {"val_test_shared_documents": shared}


def token_counts(language: str, root: Path = Path(".")) -> dict:
    """Read the measured token counts for one language.

    Args:
        language: ``"hindi"`` (the only language).
        root: Repository root.

    Returns:
        Per-split token counts plus the total.
    """
    report = json.loads(
        (root / "reports" / "data" / f"export_{language}.json").read_text())
    counts = {s["split"]: s["tokens"] for s in report["splits"]}
    counts["total"] = sum(counts.values())
    counts["manual_share"] = {s["split"]: s["manual_share"]
                              for s in report["splits"]}
    return counts


def run(language: str, sample_every: int = 20, root: Path = Path(".")) -> dict:
    """Measure token counts and split contamination for one language.

    Args:
        language: ``"hindi"`` (the only language).
        sample_every: Near-duplicate sampling stride over train.
        root: Repository root.

    Returns:
        The full report.
    """
    clean_dir = root / language / "data" / "clean"
    hasher = MinHasher()

    LOGGER.info("=== %s ===", language)
    index = index_heldout(clean_dir, hasher)
    result = scan_train(clean_dir, index, hasher, sample_every)
    result.update(val_test_overlap(clean_dir))
    result["heldout_documents"] = index["counts"]
    result["tokens"] = token_counts(language, root)
    result["near_duplicate_threshold"] = NEAR_DUPLICATE_THRESHOLD
    result["sample_every"] = sample_every
    result["clean"] = (result["exact_overlap_count"] == 0
                       and result["near_duplicate_count"] == 0
                       and result["val_test_shared_documents"] == 0)
    return result


def main() -> None:
    """Report token counts and split overlap for both languages."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--sample-every", type=int, default=20,
                        help="near-duplicate check runs on every Nth train doc")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/split_overlap.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    languages = ("hindi",)
    reports = {lang: run(lang, args.sample_every) for lang in languages}

    LOGGER.info("")
    LOGGER.info("%-8s %-6s %15s %10s", "lang", "split", "tokens", "manual%")
    LOGGER.info("-" * 46)
    for lang, rep in reports.items():
        for split in SPLITS:
            LOGGER.info("%-8s %-6s %15s %9.2f%%", lang, split,
                        f"{rep['tokens'][split]:,}",
                        100 * rep["tokens"]["manual_share"][split])
        LOGGER.info("%-8s %-6s %15s", lang, "TOTAL",
                    f"{rep['tokens']['total']:,}")

    LOGGER.info("")
    LOGGER.info("%-8s %12s %12s %12s %s", "lang", "exact", "near-dup",
                "val/test", "verdict")
    LOGGER.info("-" * 60)
    for lang, rep in reports.items():
        LOGGER.info("%-8s %12d %12d %12d %s", lang,
                    rep["exact_overlap_count"], rep["near_duplicate_count"],
                    rep["val_test_shared_documents"],
                    "CLEAN" if rep["clean"] else "CONTAMINATED")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
