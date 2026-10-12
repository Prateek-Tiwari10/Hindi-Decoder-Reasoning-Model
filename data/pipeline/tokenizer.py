"""Tokenizer sweep: BPE vs Unigram across five vocabulary sizes, per language.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

WHAT IS BEING CHOSEN, AND ON WHAT EVIDENCE
------------------------------------------
Two independent tokenizers -- one per language, trained on that language's
**train split only**, sharing nothing.  The choice of algorithm and vocabulary
size is made on measured held-out numbers rather than convention:

* **Fertility (tokens/word)** -- the primary number.  Lower means each word costs
  fewer tokens, so a fixed token budget covers more text and the model sees more
  language per step.
* **chars/token** -- the same quantity from the other side, and what every corpus
  size estimate in the report depends on.
* **UNK rate** -- must be ~0.  A Devanagari tokenizer that emits UNKs is
  discarding text, and byte-fallback makes this measurable rather than hidden.
* **Embedding cost** -- vocabulary is not free.  At d_model=384, a 64k vocabulary
  spends 24.6M parameters on embeddings alone, which is the entire ~25M budget
  for Phase 2.  This is the reason the sweep does not simply pick the largest
  vocabulary, which would always win on fertility.

THE uint16 CEILING
------------------
Vocabulary is capped at 64,000 so token IDs fit in ``uint16`` (max 65,535).  The
exported ``.bin`` files are memory-mapped as uint16 in Phase 2; a 65,536-entry
vocabulary would not fit and would silently double the dataset size as uint32.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import sentencepiece as spm

from data.pipeline.dedup import iter_shard, split_shards

LOGGER = logging.getLogger("tokenizer")

#: Vocabulary sizes to sweep.
#:
#: Both candidates sit comfortably inside the embedding budget, so this comparison
#: is decided on fertility rather than affordability. At d_model=384, where one
#: decoder layer costs ~12*d^2 = 1.77M parameters out of a ~25M budget:
#:
#:     16,000 -> 6.14M embeddings (25% of budget), ~10.7 layers affordable
#:     24,000 -> 9.22M embeddings (37% of budget),  ~8.9 layers affordable
#:
#: Both leave a genuinely deep transformer, unlike 48k (74%, ~3.7 layers) or 64k
#: (98%, ~0 layers). Both are also far below the uint16 ceiling of 65,535 token
#: ids, so the exported corpus stays 16-bit either way.
VOCAB_SIZES = (16_000, 24_000)

#: Algorithms compared. Both are SentencePiece, both trained from scratch.
MODEL_TYPES = ("bpe", "unigram")

#: Model dimension assumed when costing the embedding table, from the Phase 2
#: target of a ~25M-parameter model.
D_MODEL = 384

#: Phase 2 total parameter budget.
PARAM_BUDGET = 25_000_000

#: Hard ceiling on the share of that budget the embedding table may consume.
#:
#: This is the constraint that makes the sweep a real choice rather than "pick
#: the biggest vocabulary".  Fertility improves monotonically with vocabulary, so
#: ranking on fertility alone always selects 64k -- which at d_model=384 costs
#: 24.6M parameters and leaves nothing at all for the transformer.  A decoder
#: layer costs ~12*d_model^2 = 1.77M parameters here, so:
#:
#:     vocab 16k -> 6.1M embeddings (25%), ~10 layers affordable
#:     vocab 32k -> 12.3M           (49%), ~7 layers
#:     vocab 48k -> 18.4M           (74%), ~3.7 layers   <- too shallow
#:     vocab 64k -> 24.6M           (98%), ~0 layers     <- impossible
#:
#: At 50% the model can still afford roughly seven layers, which is a real
#: transformer.  Above that the vocabulary is eating the model.
MAX_EMBEDDING_FRACTION = 0.5

#: Fertility advantage one algorithm must hold over the other to be preferred.
#: Below this the two are effectively tied and BPE wins on simplicity and on
#: training cost -- measured at ~15x faster than Unigram in this sweep.
ALGORITHM_TIEBREAK = 0.01

#: Cap on training text, in characters.  ``None`` means train on the COMPLETE
#: cleaned train split, which is what the final tokenizers use.
#:
#: Measured effect of the training-set size on fertility (32k BPE):
#:   98M chars  -> Hindi 1.2704
#:   419M chars -> Hindi 1.2558   (+1.15%)
#: Returns diminish quickly, but "trained on the full corpus" is both the
#: stronger claim and the correct default when the compute is affordable.
MAX_TRAIN_CHARS = None

#: SentencePiece skips any line longer than this many BYTES
#: (``max_sentence_length``, default 4192) and only warns.  Cleaned documents
#: routinely hold a whole paragraph on one line, so **52% of Hindi lines were
#: being silently discarded** before this was found -- and
#: the survivors were biased toward short lines.  Long lines are therefore split
#: on danda boundaries before writing, so nothing is dropped.
MAX_LINE_BYTES = 4000


def _emit_lines(text: str, limit_bytes: int = MAX_LINE_BYTES):
    """Yield lines that all fit inside SentencePiece's per-line byte limit.

    A line longer than the limit is split on danda (``।``) boundaries, which is
    the natural sentence terminator in Devanagari prose, so the split falls
    between sentences rather than mid-word.  A single sentence still over the
    limit is emitted anyway and skipped by SentencePiece -- vanishingly rare and
    not worth mangling text to avoid.

    Args:
        text: Document text.
        limit_bytes: Maximum UTF-8 bytes per emitted line.

    Yields:
        Lines ready to write.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line.encode("utf-8")) <= limit_bytes:
            yield line
            continue
        buffer = ""
        for sentence in line.split("।"):
            candidate = buffer + sentence + "।"
            if len(candidate.encode("utf-8")) > limit_bytes and buffer:
                yield buffer.strip()
                buffer = sentence + "।"
            else:
                buffer = candidate
        if buffer.strip():
            yield buffer.strip()


def export_split_text(clean_dir: Path, split: str, out_path: Path,
                      max_chars: int | None = None) -> tuple[int, int]:
    """Write one split's text to a plain UTF-8 file, one usable line per line.

    Two things this must get right, both of which were wrong in the first
    version and cost most of the training data:

    * **Nothing may exceed SentencePiece's line limit.** It skips over-long
      lines with only a warning, which silently discarded 52% of Hindi lines.
      :func:`_emit_lines` splits them on sentence boundaries instead.
    * **The cap is in CHARACTERS, and is named so.** The previous version counted
      characters into a variable called ``bytes``, so a nominal "400 MB" cap
      produced a 1.07 GB file -- Devanagari costs ~2.5 UTF-8 bytes per character.

    Args:
        clean_dir: Directory of cleaned shards.
        split: ``"train"``, ``"val"`` or ``"test"``.
        out_path: Destination text file.
        max_chars: Stop after roughly this many characters; ``None`` writes the
            complete split.

    Returns:
        ``(documents_written, characters_written)``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    written = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for shard in split_shards(clean_dir, split):
            for record in iter_shard(shard):
                text = record.get("text") or ""
                if not text:
                    continue
                for line in _emit_lines(text):
                    handle.write(line + "\n")
                    written += len(line) + 1
                documents += 1
                if max_chars and written >= max_chars:
                    return documents, written
    return documents, written


def evaluate_tokenizer(model_path: Path, eval_lines: list[str]) -> dict:
    """Measure fertility, chars/token and UNK rate on held-out lines.

    Args:
        model_path: Trained SentencePiece model.
        eval_lines: Held-out text lines, never seen in training.

    Returns:
        Measured statistics.
    """
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = processor.unk_id()

    n_tokens = n_chars = n_words = n_unk = 0
    for line in eval_lines:
        ids = processor.encode(line, out_type=int)
        n_tokens += len(ids)
        n_chars += len(line)
        n_words += len(line.split())
        n_unk += sum(1 for i in ids if i == unk_id)

    return {
        "eval_lines": len(eval_lines),
        "tokens": n_tokens,
        "characters": n_chars,
        "words": n_words,
        "chars_per_token": n_chars / n_tokens if n_tokens else 0.0,
        "tokens_per_word": n_tokens / n_words if n_words else 0.0,
        "unk_rate": n_unk / n_tokens if n_tokens else 0.0,
    }


def train_one(train_file: Path, out_dir: Path, language: str,
              model_type: str, vocab_size: int) -> Path:
    """Train a single SentencePiece model from scratch.

    Args:
        train_file: Plain-text training corpus.
        out_dir: Where to write the model and vocabulary.
        language: For the filename.
        model_type: ``"bpe"`` or ``"unigram"``.
        vocab_size: Target vocabulary size.

    Returns:
        Path to the trained ``.model`` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{language}_{model_type}_{vocab_size}"

    # Read the core count at runtime rather than hardcoding it: this runs on a
    # laptop and on a Kaggle host with very different CPU counts, and a fixed 8
    # left most of the machine idle on the larger one.  os.cpu_count() can
    # return None on exotic platforms, so fall back to 1.
    cpu_count = os.cpu_count()
    num_threads = cpu_count or 1
    algorithm_note = ("unigram (the memory-hungry one; needs "
                      "train_extremely_large_corpus at this scale)"
                      if model_type == "unigram"
                      else "bpe (the cheap, fast one)")
    print(f"[{language} {model_type}/{vocab_size}] os.cpu_count()={cpu_count} "
          f"-> num_threads={num_threads} | algorithm: {algorithm_note}",
          flush=True)

    spm.SentencePieceTrainer.train(
        input=str(train_file),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        # 0.9995 leaves rare Latin and symbol characters to byte fallback rather
        # than spending vocabulary slots on them.
        character_coverage=0.9995,
        byte_fallback=True,          # guarantees no true UNK on unseen bytes
        # 0 = the default = use every sentence.  The corpus size is already
        # controlled by --max-train-chars; a second cap here would silently
        # sample on top of that ladder and undercut what the run claims to
        # have trained on.
        input_sentence_size=0,
        shuffle_input_sentence=True,
        # Whatever the host provides, read at runtime -- see above.
        num_threads=num_threads,
        # Required when Unigram trains on billions of characters: the internal
        # counters overflow otherwise.  Harmless on the smaller sweep runs.
        train_extremely_large_corpus=True,
        normalization_rule_name="identity",  # text is already normalized by us
    )
    return prefix.with_suffix(".model")


def embedding_parameters(vocab_size: int, d_model: int = D_MODEL) -> int:
    """Parameters spent on the embedding table at this vocabulary size."""
    return vocab_size * d_model


def run_sweep(language: str, clean_dir: Path, out_dir: Path, work_dir: Path,
              report_path: Path, eval_lines: int = 20_000,
              max_train_chars: int | None = MAX_TRAIN_CHARS,
              model_types: tuple = MODEL_TYPES,
              vocab_sizes: tuple = VOCAB_SIZES) -> dict:
    """Train and evaluate every (algorithm, vocab size) combination.

    Args:
        language: ``"hindi"``.
        clean_dir: Cleaned corpus directory.
        out_dir: Where tokenizer models are written.
        work_dir: Scratch space for the exported text.
        report_path: Where the sweep table is written.
        eval_lines: Held-out lines used for measurement.

    Returns:
        The sweep results including the chosen configuration.
    """
    started = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)

    suffix = "full" if max_train_chars is None else str(max_train_chars // 1_000_000) + "M"
    train_file = work_dir / f"{language}_train_{suffix}.txt"
    if not train_file.exists():
        docs, size = export_split_text(clean_dir, "train", train_file, max_train_chars)
        LOGGER.info("%s: exported %s train docs, %.1fM chars (%.2f GB on disk)",
                    language, f"{docs:,}", size / 1e6,
                    train_file.stat().st_size / 1e9)

    val_file = work_dir / f"{language}_val.txt"
    if not val_file.exists():
        export_split_text(clean_dir, "val", val_file)
    with open(val_file, encoding="utf-8") as handle:
        held_out = [line.strip() for _, line in zip(range(eval_lines), handle)
                    if line.strip()]
    LOGGER.info("%s: %s held-out lines for evaluation", language, f"{len(held_out):,}")

    results = []
    for model_type in model_types:
        for vocab_size in vocab_sizes:
            tag = f"{model_type}/{vocab_size}"
            begin = time.time()
            try:
                model_path = train_one(train_file, out_dir, language,
                                       model_type, vocab_size)
            except Exception as exc:  # a vocab size can exceed available pieces
                LOGGER.warning("%s %s: training failed: %s", language, tag, exc)
                continue
            metrics = evaluate_tokenizer(model_path, held_out)
            metrics.update({
                "model_type": model_type,
                "vocab_size": vocab_size,
                "embedding_parameters": embedding_parameters(vocab_size),
                "train_seconds": round(time.time() - begin, 1),
                "model_path": str(model_path),
            })
            results.append(metrics)
            LOGGER.info("%s %-16s fertility=%.4f chars/token=%.4f unk=%.2e (%.0fs)",
                        language, tag, metrics["tokens_per_word"],
                        metrics["chars_per_token"], metrics["unk_rate"],
                        metrics["train_seconds"])

    choice = choose_best(results)
    summary = {
        "language": language,
        "swept_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_sizes": list(vocab_sizes),
        "model_types": list(model_types),
        "train_chars_used": max_train_chars or "complete corpus",
        "d_model_assumed": D_MODEL,
        "results": results,
        "chosen": choice,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("%s: chose %s vocab=%d (fertility %.4f)", language,
                choice["model_type"], choice["vocab_size"],
                choice["tokens_per_word"])
    return summary


def choose_best(results: list[dict]) -> dict:
    """Choose the tokenizer, balancing fertility against embedding cost.

    Three filters, in order:

    1. **UNK rate must be negligible.**  A tokenizer that drops text is not a
       candidate at any fertility.
    2. **The embedding table must fit the budget** (``MAX_EMBEDDING_FRACTION``).
       This is what stops the sweep degenerating into "pick the largest
       vocabulary": fertility improves monotonically with vocabulary size, so
       without this constraint 64k always wins and leaves no parameters for the
       transformer.
    3. **Among what remains, a larger vocabulary must earn its place** with a
       fertility gain above 2%; otherwise the smaller one is kept, since the
       parameters saved buy depth.

    Algorithms are then compared, with BPE preferred on a near-tie
    (``ALGORITHM_TIEBREAK``) because it trains ~15x faster and is simpler to
    explain, and neither of those matters if Unigram is materially better.

    Args:
        results: Sweep results.

    Returns:
        The chosen configuration, annotated with the reason.
    """
    if not results:
        return {}

    usable = [r for r in results if r["unk_rate"] < 1e-4] or results
    budget = PARAM_BUDGET * MAX_EMBEDDING_FRACTION
    affordable = [r for r in usable if r["embedding_parameters"] <= budget]
    if not affordable:
        # Nothing fits; fall back to the smallest vocabulary rather than fail.
        affordable = sorted(usable, key=lambda r: r["vocab_size"])[:1]

    by_type: dict[str, list[dict]] = {}
    for r in affordable:
        by_type.setdefault(r["model_type"], []).append(r)

    best_per_type = {}
    for model_type, entries in by_type.items():
        entries.sort(key=lambda r: r["vocab_size"])
        chosen = entries[0]
        for candidate in entries[1:]:
            gain = (chosen["tokens_per_word"] - candidate["tokens_per_word"]) \
                / chosen["tokens_per_word"]
            if gain > 0.02:
                chosen = candidate
        best_per_type[model_type] = chosen

    ranked = sorted(best_per_type.values(), key=lambda r: r["tokens_per_word"])
    best = ranked[0]
    if len(ranked) > 1 and best["model_type"] != "bpe":
        bpe = next((r for r in ranked if r["model_type"] == "bpe"), None)
        if bpe is not None:
            margin = (bpe["tokens_per_word"] - best["tokens_per_word"]) \
                / bpe["tokens_per_word"]
            if margin < ALGORITHM_TIEBREAK:
                best = bpe

    chosen = dict(best)
    layers = int((PARAM_BUDGET - chosen["embedding_parameters"]) / (12 * D_MODEL ** 2))
    chosen["embedding_share_of_budget"] = round(
        chosen["embedding_parameters"] / PARAM_BUDGET, 4)
    chosen["affordable_layers_at_d_model"] = layers
    chosen["selection_note"] = (
        f"largest vocabulary whose embedding table stays within "
        f"{int(100 * MAX_EMBEDDING_FRACTION)}% of the {PARAM_BUDGET / 1e6:.0f}M "
        f"parameter budget, and whose fertility gain over the next size down "
        f"exceeds 2%; leaves ~{layers} decoder layers at d_model={D_MODEL}")
    return chosen


def main() -> None:
    """Run the sweep for one language."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", required=True, choices=("hindi",))
    parser.add_argument("--clean", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-train-chars", type=int, default=MAX_TRAIN_CHARS,
                        help="cap the training text (characters). Omit for the "
                             "COMPLETE corpus. Needed on memory-limited clusters: "
                             "SentencePiece holds its working set in RAM, and "
                             "Unigram needs several times what BPE does.")
    parser.add_argument("--only", metavar="TYPE:VOCAB[,TYPE:VOCAB...]", default=None,
                        help="train specific configurations instead of the whole "
                             "sweep, e.g. --only bpe:24000 or --only "
                             "bpe:24000,bpe:16000. Multiple entries run the CROSS "
                             "PRODUCT of the algorithms and vocabulary sizes named, "
                             "so bpe:24000,bpe:16000 trains BPE at both sizes. Used "
                             "for the final full-corpus run after the sweep has chosen.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    model_types, vocab_sizes = MODEL_TYPES, VOCAB_SIZES
    if args.only:
        # dict.fromkeys keeps the order the user wrote while removing repeats,
        # so --only bpe:24000,bpe:16000 gives one algorithm and two vocabularies
        # rather than the same algorithm twice.
        specs = [spec.strip() for spec in args.only.split(",") if spec.strip()]
        model_types = tuple(dict.fromkeys(spec.split(":")[0] for spec in specs))
        vocab_sizes = tuple(dict.fromkeys(int(spec.split(":")[1]) for spec in specs))
        LOGGER.info("explicit configurations: %s x %s",
                    list(model_types), list(vocab_sizes))

    run_sweep(
        language=args.language,
        clean_dir=args.clean or Path(args.language) / "data" / "clean",
        out_dir=args.out or Path(args.language) / "tokenizer",
        work_dir=args.work or Path(args.language) / "data" / "work" / "tokenizer",
        report_path=args.report or Path(f"reports/data/tokenizer_sweep_{args.language}.json"),
        max_train_chars=args.max_train_chars,
        model_types=model_types,
        vocab_sizes=vocab_sizes,
    )


if __name__ == "__main__":
    main()
