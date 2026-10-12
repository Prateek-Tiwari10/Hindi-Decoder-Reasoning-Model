"""Memory-mapped token streams and the batch sampler.

WHY MEMMAP
----------
Each ``train.bin`` is a flat ``uint16`` array of roughly 500M token ids -- about
1 GB on disk.  ``np.memmap`` hands out a array-like view backed by the page
cache, so a batch read touches only the pages it needs and the corpus is never
resident in RAM.  This is what makes the training loop indifferent to corpus
size.

WHY RANDOM START INDICES OVER THE WHOLE FILE
--------------------------------------------
The sampler draws a uniformly random start offset across the *entire* file
rather than walking it in order.  Two consequences the plan leans on:

  1. The corpus is a concatenation of many sources in write order.  Sequential
     reading would train on one source at a time -- a curriculum nobody chose.
     Random offsets make every batch a representative mixture.
  2. It is what lets a short LR-sweep run (40M tokens, under a tenth of the
     corpus) stand in for the full run: the sweep sees the same *distribution*
     as production, just less of it.

WHY NOT DOCUMENT-ALIGNED BATCHES
--------------------------------
Windows are cut from the raw stream and may straddle the EOS separating two
documents.  That is the standard arrangement and it is deliberate: the model
learns that EOS ends a document, which is exactly what generation needs in
Part 6.  The cost is that a few windows contain a topic switch, which is noise
the model learns to condition on rather than a bug.

TRAIN / VAL / TEST
------------------
``val.bin`` and ``test.bin`` come from Phase 1's held-out splits, which were
separated *before* cleaning and which the train split was then decontaminated
against.  Nothing here re-splits anything; keeping the split boundary in Phase 1
is what makes the Part 5 numbers meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from model.config import Paths

LOGGER = logging.getLogger("data")

#: Phase 1 wrote the exports as uint16; ids must therefore stay below 65,536.
TOKEN_DTYPE = np.uint16

SPLITS = ("train", "val", "test")


class TokenStream:
    """A read-only memory-mapped ``uint16`` token stream for one split.

    Attributes:
        path: The ``.bin`` file backing this stream.
        context_len: Window length the sampler cuts.
    """

    def __init__(self, path: Path, context_len: int) -> None:
        """Open a token stream.

        Args:
            path: Path to a ``uint16`` ``.bin`` file.
            context_len: Sequence length the model consumes.

        Raises:
            FileNotFoundError: If the file is missing.
            ValueError: If the file is too short to yield a single window.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} does not exist -- run data.pipeline.export first")
        self.context_len = int(context_len)
        self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        # A window needs context_len inputs plus the one target that follows
        # the last input, hence the +1.
        if len(self._tokens) < self.context_len + 1:
            raise ValueError(
                f"{self.path} holds {len(self._tokens)} tokens, too few for a "
                f"{self.context_len}-token window")

    def __len__(self) -> int:
        """Number of tokens in the stream."""
        return len(self._tokens)

    @property
    def n_windows(self) -> int:
        """Number of distinct start offsets the sampler may draw."""
        return len(self._tokens) - self.context_len

    @property
    def tokens(self) -> np.memmap:
        """The underlying memmap, for callers that need raw ids."""
        return self._tokens

    def window(self, start: int) -> tuple[np.ndarray, np.ndarray]:
        """Cut one ``(x, y)`` pair at a given offset.

        ``y`` is ``x`` shifted left by one: the target at position ``t`` is the
        token the model should predict having seen everything up to ``t``.  Both
        come from a single ``context_len + 1`` slice, so they cannot drift out
        of alignment.

        Args:
            start: Index of the first input token.

        Returns:
            ``(x, y)``, each of shape ``(context_len,)`` and dtype int64.
        """
        block = np.asarray(
            self._tokens[start:start + self.context_len + 1], dtype=np.int64)
        return block[:-1], block[1:]

    def sample(self, batch_size: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a batch of windows from uniformly random offsets.

        Args:
            batch_size: Number of sequences in the batch.
            rng: Source of randomness.  Pass a seeded generator to make an
                evaluation batch reproducible across steps and runs.

        Returns:
            ``(x, y)``, each of shape ``(batch_size, context_len)``, int64.
        """
        starts = rng.integers(0, self.n_windows, size=batch_size)
        pairs = [self.window(int(s)) for s in starts]
        return (np.stack([p[0] for p in pairs]),
                np.stack([p[1] for p in pairs]))

    def iter_blocks(self, batch_size: int, stride: int | None = None
                    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Walk the stream in order with non-overlapping windows.

        Random sampling is right for training but wrong for evaluation: a
        perplexity computed over random windows would score some tokens twice
        and others never.  Part 5 needs each token scored exactly once, which
        is what a sequential, non-overlapping sweep gives.

        Args:
            batch_size: Sequences per yielded batch.
            stride: Distance between window starts.  Defaults to
                ``context_len`` (no overlap, no gaps).

        Yields:
            ``(x, y)`` batches; the final batch may be smaller.
        """
        stride = stride or self.context_len
        starts = range(0, self.n_windows, stride)
        batch: list[tuple[np.ndarray, np.ndarray]] = []
        for start in starts:
            batch.append(self.window(start))
            if len(batch) == batch_size:
                yield (np.stack([b[0] for b in batch]),
                       np.stack([b[1] for b in batch]))
                batch = []
        if batch:
            yield (np.stack([b[0] for b in batch]),
                   np.stack([b[1] for b in batch]))


def to_torch(batch: tuple[np.ndarray, np.ndarray], device: str = "cpu"):
    """Move one ``(x, y)`` numpy batch onto a torch device.

    Kept separate from :class:`TokenStream` so the sampler can be tested, and
    the token counts verified, without importing torch at all.

    Args:
        batch: ``(x, y)`` arrays from :meth:`TokenStream.sample`.
        device: Torch device string.

    Returns:
        ``(x, y)`` as int64 tensors on ``device``.  CUDA transfers are
        pinned and non-blocking so the copy overlaps the previous step.
    """
    import torch

    x, y = (torch.from_numpy(np.ascontiguousarray(a)) for a in batch)
    if device.startswith("cuda"):
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


def file_digest(path: Path, limit: int = 64 << 20) -> str:
    """Hash a file's head, as a cheap identity check.

    A full hash of a 1 GB memmap costs seconds on every startup for no extra
    confidence; the first 64 MB is more than enough to notice that a ``.bin``
    was rebuilt with a different tokenizer.

    Args:
        path: File to digest.
        limit: Maximum bytes to read.

    Returns:
        Hex SHA-1 of the first ``limit`` bytes.
    """
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def write_sidecar(language: str, split: str, root: Path = Path("."),
                  variant: str = "16k") -> dict:
    """Measure one ``.bin`` and write its sidecar JSON.

    The sidecar records what the memmap alone cannot say: how many tokens it
    holds, which tokenizer produced them, and the highest id present.  Training
    reads it to assert that the ``.bin`` on disk matches the vocabulary the
    model was configured for -- the exact mismatch that silently corrupts a run.

    Args:
        language: ``"hindi"`` (the only language).
        split: One of :data:`SPLITS`.
        root: Repository root.
        variant: Architecture variant, which selects the tokenizer and bins.

    Returns:
        The sidecar contents.
    """
    paths = Paths.for_variant(language, root, variant)
    bin_path = paths.split_bin(split)
    tokens = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")

    sidecar = {
        "language": language,
        "variant": variant,
        "vocab_size": paths.vocab_size,
        "split": split,
        "path": str(bin_path),
        "tokens": int(len(tokens)),
        "bytes": bin_path.stat().st_size,
        "dtype": "uint16",
        "max_token_id": int(tokens.max()),
        "min_token_id": int(tokens.min()),
        "sha1_head_64mb": file_digest(bin_path),
        "tokenizer": str(paths.tokenizer),
        "tokenizer_md5": hashlib.md5(
            paths.tokenizer.read_bytes()).hexdigest(),
    }
    out = bin_path.with_suffix(".json")
    out.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar


def verify(language: str, context_len: int = 512, root: Path = Path("."),
           n_preview: int = 2, variant: str = "16k") -> dict:
    """Run the Part 1 verification checks for one language.

    Checks, in order: all three splits exist and are named correctly; every id
    is inside the vocabulary; ``y`` really is ``x`` shifted by one; and a
    sampled window decodes back to readable text in the right language.

    Args:
        language: ``"hindi"`` (the only language).
        context_len: Window length to sample.
        root: Repository root.
        n_preview: How many decoded samples to include.
        variant: Architecture variant, which selects the tokenizer and bins.
            The expected vocabulary comes from it rather than from a separate
            argument, so the two cannot be given inconsistently.

    Returns:
        A report mapping each split to its sidecar plus the checks' outcomes.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, variant)
    processor = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
    rng = np.random.default_rng(0)

    report: dict = {"language": language,
                    "variant": variant,
                    "tokenizer": str(paths.tokenizer),
                    "tokenizer_vocab": processor.get_piece_size(),
                    "expected_vocab": paths.vocab_size,
                    "splits": {}}

    for split in SPLITS:
        sidecar = write_sidecar(language, split, root, variant)
        stream = TokenStream(paths.split_bin(split), context_len)
        x, y = stream.sample(4, rng)

        # y[t] must be the token following x[t]; equivalently the two arrays
        # overlap everywhere but their ends.
        shift_ok = bool(np.array_equal(x[:, 1:], y[:, :-1]))

        checks = {
            "ids_within_vocab": sidecar["max_token_id"] < paths.vocab_size,
            "y_is_x_shifted_by_one": shift_ok,
            "shapes": [list(x.shape), list(y.shape)],
        }
        sidecar["checks"] = checks
        sidecar["decoded_samples"] = [
            processor.decode(row.tolist())[:200] for row in x[:n_preview]]
        report["splits"][split] = sidecar

    total = sum(s["tokens"] for s in report["splits"].values())
    report["total_tokens"] = total
    report["all_checks_passed"] = all(
        s["checks"]["ids_within_vocab"] and s["checks"]["y_is_x_shifted_by_one"]
        for s in report["splits"].values())
    return report


def main() -> None:
    """Verify the exported token streams and write their sidecars."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/data.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    languages = ("hindi",)
    reports = {}
    for language in languages:
        report = verify(language, context_len=args.context_len,
                        root=args.root, variant=args.variant)
        reports[language] = report
        LOGGER.info("\n=== %s ===", language)
        for split, s in report["splits"].items():
            LOGGER.info("  %-5s %15s tokens  max_id=%5d  %s",
                        split, f"{s['tokens']:,}", s["max_token_id"],
                        "OK" if all(v is True for k, v in s["checks"].items()
                                    if isinstance(v, bool)) else "FAILED")
        LOGGER.info("  sample: %s",
                    report["splits"]["train"]["decoded_samples"][0][:120])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("\nreport -> %s", args.report)


if __name__ == "__main__":
    main()
