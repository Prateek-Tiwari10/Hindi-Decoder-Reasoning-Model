"""Assemble the final corpus and export it as uint16 ``.bin`` memmaps for Phase 2.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

ASSEMBLY, NOT JUST TOKENIZATION
-------------------------------
The cleaned corpus is larger than the 500M-token target -- deliberately, since
cleaning can only remove tokens and the crawl window has closed.  This module
therefore *assembles* the training corpus rather than tokenizing everything:

  1. Keep **every** manual document.  They are unreplaceable and they are what
     the >=20% manual requirement is measured on.
  2. Subsample public documents to fill the remainder up to the target.

That ordering is what makes a large public pool an asset: the manual share of
the final corpus is chosen rather than inherited.  Subsampling is by document
hash, so it is deterministic and reproducible -- rerunning the export produces
byte-identical output.

WHY uint16
----------
Token IDs fit in 16 bits for any vocabulary up to 65,535, so a 500M-token corpus
is a 1.0 GB file that Phase 2 memory-maps directly.  uint32 would double that
for no benefit, which is the reason the sweep caps vocabulary at 64,000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

from data.pipeline.dedup import iter_shard, split_shards

LOGGER = logging.getLogger("export")

#: Held-out splits are exported whole; only train is subsampled.
UNSAMPLED_SPLITS = ("val", "test")


def keep_public_document(text: str, keep_fraction: float) -> bool:
    """Decide deterministically whether to keep one public document.

    Uses a hash of the text rather than a random draw so the decision is stable
    across runs and machines: the exported corpus must be reproducible.

    Args:
        text: Document text.
        keep_fraction: Fraction of public documents to retain.

    Returns:
        True if the document should be kept.
    """
    if keep_fraction >= 1.0:
        return True
    # A different hash prefix than the split assignment uses, so subsampling is
    # independent of which split a document landed in.
    digest = hashlib.sha1(("subsample:" + text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < keep_fraction


def export_split(clean_dir: Path, split: str, processor: spm.SentencePieceProcessor,
                 out_path: Path, public_keep_fraction: float = 1.0) -> dict:
    """Tokenize one split and write it as a flat uint16 array.

    Documents are separated by the tokenizer's EOS id, so Phase 2 can find
    document boundaries in a stream that is otherwise a flat token array.

    Args:
        clean_dir: Cleaned corpus directory.
        split: Split name.
        processor: Trained tokenizer.
        out_path: Destination ``.bin`` file.
        public_keep_fraction: Fraction of public documents to keep.

    Returns:
        Counts for the report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eos = processor.eos_id()
    if eos < 0:
        eos = processor.piece_to_id("</s>")

    n_tokens = 0
    n_docs = 0
    manual_tokens = 0
    public_tokens = 0
    skipped_public = 0
    started = time.time()

    # Write incrementally in chunks so peak memory stays flat regardless of
    # corpus size; the file is a plain flat array Phase 2 can np.memmap.
    buffer: list[int] = []
    with open(out_path, "wb") as handle:
        for shard in split_shards(clean_dir, split):
            for record in iter_shard(shard):
                text = record.get("text") or ""
                if not text:
                    continue
                is_manual = (record.get("source_type") == "manual")
                if not is_manual and not keep_public_document(text, public_keep_fraction):
                    skipped_public += 1
                    continue

                ids = processor.encode(text, out_type=int)
                ids.append(eos)
                buffer.extend(ids)
                n_tokens += len(ids)
                n_docs += 1
                if is_manual:
                    manual_tokens += len(ids)
                else:
                    public_tokens += len(ids)

                if len(buffer) >= 8_000_000:
                    np.asarray(buffer, dtype=np.uint16).tofile(handle)
                    buffer.clear()
                    LOGGER.info("  %s: %s tokens (%.0fs)", split,
                                f"{n_tokens:,}", time.time() - started)
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(handle)

    return {
        "split": split,
        "documents": n_docs,
        "tokens": n_tokens,
        "manual_tokens": manual_tokens,
        "public_tokens": public_tokens,
        "manual_share": round(manual_tokens / n_tokens, 4) if n_tokens else 0.0,
        "public_documents_skipped": skipped_public,
        "public_keep_fraction": public_keep_fraction,
        "bytes_on_disk": out_path.stat().st_size,
        "path": str(out_path),
    }


def verify_export(path: Path, vocab_size: int, expected_tokens: int) -> dict:
    """Read the exported file back and confirm it is well-formed.

    Catches the two failure modes that would silently corrupt Phase 2: a token id
    outside the vocabulary (which means the uint16 assumption broke) and a token
    count that disagrees with what was written.

    Args:
        path: Exported ``.bin`` file.
        vocab_size: Tokenizer vocabulary size.
        expected_tokens: Token count reported by the export.

    Returns:
        Verification result.
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    max_id = int(data.max()) if data.size else 0
    return {
        "tokens_on_disk": int(data.size),
        "expected_tokens": expected_tokens,
        "count_matches": int(data.size) == expected_tokens,
        "max_token_id": max_id,
        "vocab_size": vocab_size,
        "ids_within_vocab": max_id < vocab_size,
        "dtype": "uint16",
    }


def run(language: str, clean_dir: Path, tokenizer_report: Path, out_dir: Path,
        report_path: Path, target_tokens: int, audit_report: Path | None) -> dict:
    """Assemble, tokenize, export and verify one language.

    Args:
        language: ``"hindi"``.
        clean_dir: Cleaned corpus directory.
        tokenizer_report: Sweep report naming the chosen tokenizer.
        out_dir: Destination for ``.bin`` files.
        report_path: Where to write export statistics.
        target_tokens: Target training-token count.
        audit_report: Audit report supplying the public keep fraction.

    Returns:
        Export statistics.
    """
    started = time.time()
    sweep = json.loads(tokenizer_report.read_text())
    chosen = sweep["chosen"]
    processor = spm.SentencePieceProcessor(model_file=chosen["model_path"])
    LOGGER.info("%s: using %s vocab=%d", language,
                chosen["model_type"], chosen["vocab_size"])

    keep_fraction = 1.0
    if audit_report and audit_report.exists():
        plan = json.loads(audit_report.read_text()).get("assembly_plan", {})
        keep_fraction = plan.get("public_keep_fraction", 1.0) or 1.0
        LOGGER.info("%s: assembly plan keeps %.1f%% of public documents",
                    language, 100 * keep_fraction)

    splits = []
    for split in ("train",) + UNSAMPLED_SPLITS:
        fraction = keep_fraction if split == "train" else 1.0
        result = export_split(clean_dir, split, processor,
                              out_dir / f"{split}.bin", fraction)
        result["verification"] = verify_export(
            Path(result["path"]), chosen["vocab_size"], result["tokens"])
        splits.append(result)
        LOGGER.info("%s %s: %s tokens, %.1f%% manual, %s",
                    language, split, f"{result['tokens']:,}",
                    100 * result["manual_share"],
                    "verified" if result["verification"]["ids_within_vocab"] else "FAILED")

    summary = {
        "language": language,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer": {"model_type": chosen["model_type"],
                      "vocab_size": chosen["vocab_size"],
                      "model_path": chosen["model_path"]},
        "target_tokens": target_tokens,
        "splits": splits,
        "total_tokens": sum(s["tokens"] for s in splits),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", report_path)
    return summary


def main() -> None:
    """Export one language."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True, choices=("hindi",))
    parser.add_argument("--clean", type=Path, default=None)
    parser.add_argument("--tokenizer-report", type=Path, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument(
        "--no-subsample", action="store_true",
        help="Keep every public document, ignoring the audit's keep fraction. "
             "Correct when the cleaned corpus is at or below the target: "
             "subsampling would then drop tokens the target needs. Note this "
             "is the WORST case for the manual share, since discarding public "
             "text only raises it -- so the 20%% floor must be re-checked "
             "against the export report.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    run(language=args.language,
        clean_dir=args.clean or Path(args.language) / "data" / "clean",
        tokenizer_report=(args.tokenizer_report or
                          Path(f"reports/data/tokenizer_sweep_{args.language}.json")),
        out_dir=args.out or Path(args.language) / "data" / "bin",
        report_path=args.report or Path(f"reports/data/export_{args.language}.json"),
        target_tokens=args.target_tokens,
        audit_report=(None if args.no_subsample else
                      (args.audit_report or
                       Path(f"reports/data/audit_{args.language}.json"))))


if __name__ == "__main__":
    main()
