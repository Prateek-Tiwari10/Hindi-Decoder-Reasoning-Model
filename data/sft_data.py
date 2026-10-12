"""Supervised finetuning data pipeline for the Phase 3 reasoning task.

THE ONE DECISION THAT MAKES THIS SFT RATHER THAN MORE PRETRAINING
-----------------------------------------------------------------
Loss is computed on completion tokens only.  Prompt tokens -- everything up to
and including the ``उत्तर:`` cue that ends the ``prompt`` field -- carry the
label ``-100`` and contribute nothing to the gradient.  Without that mask the
model would spend most of its capacity learning to generate *questions*, which
is not the task and is already what pretraining did.

WHY ``-100`` NEEDS NO CHANGE TO THE PHASE 2 MODEL
-------------------------------------------------
``model.gpt.DecoderLM.forward`` computes ``F.cross_entropy(logits, targets)``
with no explicit ``ignore_index``, and PyTorch's default for that argument is
exactly ``-100``.  So the masked positions are dropped from the loss, and the
reduction is the mean over the *surviving* positions -- the mean over
completion tokens, which is what we want.  The architecture is untouched, which
is both a requirement and the reason the finetune is comparable to the base
model.

WHY RIGHT PADDING IS SAFE WITHOUT AN ATTENTION MASK
---------------------------------------------------
The Phase 2 forward pass takes no padding mask, and none is added here.  It is
not needed, because attention is causal: a real token at position ``t`` can
only attend to positions ``<= t``, so padding appended *after* the real tokens
cannot influence any real position's hidden state.  The padded positions
themselves produce garbage logits, and those are excluded from the loss by
their ``-100`` labels.  Both halves of that argument are verified empirically
by gate P7 rather than asserted.

THE SHIFT
---------
Phase 2's data pipeline hands the model ``x`` and ``y`` already shifted by one
(``data.token_stream.TokenStream.window``), and ``forward`` documents that it does not
shift again.  This pipeline keeps that contract: it builds the aligned
``input_ids`` / ``labels`` pair of §2 and applies the shift in the collate, so
``x = input_ids[:-1]`` and ``y = labels[1:]``.  The consequence worth stating
is that the LAST PROMPT TOKEN is the first position with an active label: it is
the position that predicts the first completion token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

#: Label marking a position that must not contribute to the loss.  Matches
#: ``F.cross_entropy``'s default ``ignore_index``, which is why no argument has
#: to be threaded through the Phase 2 forward pass.
IGNORE_INDEX = -100

#: Safety truncation.  The generator report measured a maximum of 237 Hindi
#: tokens for ``prompt + completion`` in the first build; plus EOS that is 238.  This
#: bound should therefore never bind, and :func:`load_examples` counts it if it
#: ever does rather than silently dropping the tail of a chain of thought.
MAX_LEN = 288


@dataclass(frozen=True)
class SFTExample:
    """One finetuning example, tokenized and masked.

    Attributes:
        record_id: The dataset record's ``id``, for tracing back.
        input_ids: ``prompt_ids + completion_ids + [eos]``.
        labels: Same length as ``input_ids``; ``IGNORE_INDEX`` over the prompt
            span, the true token elsewhere.
        n_prompt: Length of the prompt span, so the boundary is recoverable
            without re-tokenizing.
        answer: The gold label, for exact-match scoring.
        slice_tag: Which slice the record came from.
    """

    record_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    n_prompt: int
    answer: str
    slice_tag: str

    @property
    def n_completion(self) -> int:
        """Number of loss-active tokens, EOS included."""
        return len(self.input_ids) - self.n_prompt

    @property
    def n_supervised(self) -> int:
        """Loss-active positions after the shift.

        The shift drops one position from the front of ``labels``, and that
        dropped position is always inside the masked prompt span, so the count
        equals :attr:`n_completion`.  Kept as its own property because it is
        the number the token-weighted gradient accumulation divides by, and
        conflating the two would be a silent off-by-one in the loss scale.
        """
        return sum(1 for label in self.labels[1:] if label != IGNORE_INDEX)


def build_example(record: dict, tokenizer, eos_id: int,
                  max_len: int = MAX_LEN) -> tuple[SFTExample, bool]:
    """Tokenize and mask one dataset record.

    ``prompt`` and ``completion`` are encoded SEPARATELY and concatenated as
    ids.  That is what makes the mask boundary exact: encoding the joined
    string and then trying to find the boundary by counting characters would be
    at the mercy of whether the tokenizer merges the last prompt token with the
    completion's leading space.

    Args:
        record: One record from a Part 1 ``.jsonl`` file.
        tokenizer: Loaded SentencePiece processor, read-only.
        eos_id: End-of-sequence id.  Appended to the completion so the model
            learns to STOP; without it generation runs to the length cap.
        max_len: Safety truncation.

    Returns:
        ``(example, truncated)``.
    """
    prompt_ids = tokenizer.encode(record["prompt"], out_type=int)
    completion_ids = tokenizer.encode(record["completion"], out_type=int)
    completion_ids = [*completion_ids, eos_id]

    input_ids = [*prompt_ids, *completion_ids]
    truncated = len(input_ids) > max_len
    if truncated:
        input_ids = input_ids[:max_len]

    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + list(input_ids[n_prompt:])

    return SFTExample(
        record_id=record["id"],
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        n_prompt=n_prompt,
        answer=record["answer"],
        slice_tag=record["slice"],
    ), truncated


def load_examples(path: Path, tokenizer, eos_id: int, max_len: int = MAX_LEN,
                  families: Sequence[str] | None = None
                  ) -> tuple[list[SFTExample], dict]:
    """Load and tokenize a whole ``.jsonl`` split.

    Args:
        path: Dataset file.
        tokenizer: Loaded SentencePiece processor.
        eos_id: End-of-sequence id.
        max_len: Safety truncation.
        families: If given, keep only these ``task_family`` values.  Used by
            the optional relation-only enrichment runs.

    Returns:
        ``(examples, stats)`` where stats reports counts, truncations and the
        token-length distribution the trainer logs.
    """
    examples: list[SFTExample] = []
    truncations = 0
    skipped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if families is not None and record["task_family"] not in families:
            skipped += 1
            continue
        example, truncated = build_example(record, tokenizer, eos_id, max_len)
        truncations += truncated
        examples.append(example)

    lengths = np.array([len(e.input_ids) for e in examples])
    supervised = np.array([e.n_completion for e in examples])
    stats = {
        "path": str(path),
        "examples": len(examples),
        "skipped_by_family": skipped,
        "truncated": int(truncations),
        "max_len": max_len,
        "tokens_total": int(lengths.sum()),
        "tokens_supervised": int(supervised.sum()),
        "length_mean": round(float(lengths.mean()), 2),
        "length_p99": int(np.percentile(lengths, 99)),
        "length_max": int(lengths.max()),
        "completion_mean": round(float(supervised.mean()), 2),
    }
    return examples, stats


def collate(batch: Sequence[SFTExample], pad_id: int,
            device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad a batch to its own longest sequence and apply the causal shift.

    Dynamic padding rather than a fixed 512: these sequences average about 100
    tokens, so padding every batch to the context length would spend roughly
    five times the compute on positions that are masked out anyway.  No
    packing, because packing several examples into one row means either a
    block-diagonal attention mask -- which the Phase 2 forward has no argument
    for -- or letting one example attend to another's tokens.

    Args:
        batch: Examples to combine.
        pad_id: Filler id for the input side.  Any id works: trailing padding
            cannot reach a real position under causal attention, and its own
            predictions are dropped by the ``-100`` label.  EOS is the sensible
            choice so a decoded padded row reads as terminated text.
        device: Torch device for the returned tensors.

    Returns:
        ``(x, y, n_supervised)`` where ``x`` is ``(B, T-1)`` inputs, ``y`` is
        ``(B, T-1)`` shifted labels, and ``n_supervised`` is the number of
        loss-active positions in the batch -- needed for token-weighted
        gradient accumulation.
    """
    width = max(len(e.input_ids) for e in batch)
    x = np.full((len(batch), width - 1), pad_id, dtype=np.int64)
    y = np.full((len(batch), width - 1), IGNORE_INDEX, dtype=np.int64)

    for row, example in enumerate(batch):
        ids = example.input_ids
        labels = example.labels
        # x drops the last token (it has nothing to predict); y drops the
        # first label (nothing predicts position 0).
        x[row, :len(ids) - 1] = ids[:-1]
        y[row, :len(labels) - 1] = labels[1:]

    n_supervised = int((y != IGNORE_INDEX).sum())
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device), n_supervised)


def epoch_order(n_examples: int, seed: int, epoch: int,
                bucket: int = 0) -> np.ndarray:
    """The example order for one epoch, as a pure function of seed and epoch.

    THIS IS WHAT MAKES MID-EPOCH RESUME EXACT.  If the order came from a
    long-lived generator advanced once per epoch, then restarting would draw
    the NEXT permutation rather than the remainder of the interrupted one, and
    the resumed run would train on different batches -- so the resumed loss
    curve could never be continuous with the pre-kill curve, and gate P4 could
    not be made to pass honestly.  Deriving the order from ``(seed, epoch)``
    means a resume can rebuild the identical order and skip the batches it
    already consumed.

    Args:
        n_examples: Size of the split.
        seed: Master seed.
        epoch: Which epoch, 0-based.
        bucket: If > 1, sort by length inside windows of this many examples
            after shuffling, so dynamic padding wastes less while the order
            stays random at the scale that matters for SGD.  The caller
            supplies lengths via :func:`batches_from_order`, so this function
            only needs the window size.

    Returns:
        A permutation of ``range(n_examples)``.
    """
    return np.random.default_rng(seed * 100_003 + epoch).permutation(n_examples)


def apply_length_buckets(examples: Sequence[SFTExample], order: np.ndarray,
                         bucket: int) -> np.ndarray:
    """Sort by sequence length inside fixed windows of a shuffled order.

    Args:
        examples: The split, for lengths.
        order: A shuffled permutation.
        bucket: Window size; values below 2 return ``order`` unchanged.

    Returns:
        The reordered permutation.
    """
    if bucket < 2:
        return order
    chunks: list[int] = []
    for start in range(0, len(order), bucket):
        window = order[start:start + bucket]
        chunks.extend(sorted(window, key=lambda i: len(examples[i].input_ids)))
    return np.array(chunks, dtype=order.dtype)


def batches_from_order(examples: Sequence[SFTExample], order: np.ndarray,
                       batch_size: int) -> list[list[SFTExample]]:
    """Cut an example order into micro-batches.

    Args:
        examples: The split.
        order: Index order to follow.
        batch_size: Micro-batch size.

    Returns:
        Micro-batches, in order.
    """
    return [[examples[i] for i in order[start:start + batch_size]]
            for start in range(0, len(order), batch_size)
            if len(order[start:start + batch_size])]


def epoch_batches(examples: Sequence[SFTExample], batch_size: int, seed: int,
                  epoch: int, bucket: int = 0) -> list[list[SFTExample]]:
    """All micro-batches for one epoch, deterministic in ``(seed, epoch)``.

    Args:
        examples: The split to iterate.
        batch_size: Micro-batch size.
        seed: Master seed.
        epoch: Which epoch, 0-based.
        bucket: Length-bucket window; 0 disables.

    Returns:
        Micro-batches, in order.
    """
    order = epoch_order(len(examples), seed, epoch, bucket)
    order = apply_length_buckets(examples, order, bucket)
    return batches_from_order(examples, order, batch_size)
