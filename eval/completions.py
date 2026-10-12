"""Greedy decoding that stops at EOS, plus exact-match scoring.

WHY NOT REUSE ``eval.generation_metrics.generate``
--------------------------------------
Phase 2's generator deliberately ignores EOS and always runs the full length,
because its diversity statistics (repetition rate, Distinct-n) are only
comparable across samples of equal length.  That is the right choice there and
the wrong one here: the whole point of appending EOS during finetuning is to
teach the model to stop, and a decoder that cannot stop cannot show whether it
learned to.  Gate P5 tests exactly that, so this module provides the
EOS-terminating variant instead of changing Phase 2's recorded behaviour.

WHY BATCHING IS BUCKETED BY PROMPT LENGTH
-----------------------------------------
The model uses LEARNED ABSOLUTE positional embeddings.  Left-padding a batch of
unequal prompts -- the usual trick for batched generation -- would place each
prompt's tokens at the wrong absolute positions and silently corrupt every
continuation.  Right-padding is no better: the generated tokens would land
after the padding rather than after the prompt.  So prompts are grouped by
EXACT token length, which batches perfectly with no padding at all and no
position shift, at the cost of several smaller batches instead of one large
one.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

# The single source of truth for the G8 rule: the same function that Part 1
# used to verify every generated label is recoverable.  Re-implementing it here
# would let the scorer and the dataset drift apart.
from data.reasoning.generator import extract_answer


@torch.no_grad()
def greedy_continue(model: torch.nn.Module, prompts: torch.Tensor,
                    max_new_tokens: int, eos_id: int) -> list[list[int]]:
    """Greedily continue a batch of equal-length prompts, stopping at EOS.

    Args:
        model: Model in eval mode.
        prompts: ``(B, T0)`` prompt ids, all rows the same real length.
        max_new_tokens: Cap on generated tokens.
        eos_id: Stop token.

    Returns:
        One list of generated ids per row, EOS excluded, truncated at the first
        EOS that row produced.
    """
    tokens = prompts.clone()
    batch = tokens.size(0)
    finished = torch.zeros(batch, dtype=torch.bool, device=tokens.device)
    generated: list[list[int]] = [[] for _ in range(batch)]

    for _ in range(max_new_tokens):
        window = tokens[:, -model.config.context_len:]
        logits, _, _ = model(window, last_position_only=True)
        next_token = logits[:, -1, :].float().argmax(dim=-1)

        for row in range(batch):
            if finished[row]:
                continue
            token = int(next_token[row])
            if token == eos_id:
                finished[row] = True
            else:
                generated[row].append(token)

        # Rows that are done keep being fed EOS so the batch stays rectangular;
        # their output is already frozen by the `finished` guard above.
        next_token = torch.where(finished, torch.full_like(next_token, eos_id),
                                 next_token)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break

    return generated


@torch.no_grad()
def generate_completions(model: torch.nn.Module, prompts: Sequence[Sequence[int]],
                         eos_id: int, device: str, max_new_tokens: int = 160,
                         max_batch: int = 64) -> list[list[int]]:
    """Generate a completion for each prompt, bucketing by length.

    Args:
        model: Model in eval mode.
        prompts: Prompt id sequences, any lengths.
        eos_id: Stop token.
        device: Torch device.
        max_new_tokens: Cap on generated tokens per prompt.
        max_batch: Largest batch to run at once.

    Returns:
        Generated ids per prompt, in the input order.
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, prompt in enumerate(prompts):
        buckets[len(prompt)].append(index)

    results: list[list[int]] = [[] for _ in prompts]
    for length in sorted(buckets):
        indices = buckets[length]
        for start in range(0, len(indices), max_batch):
            chunk = indices[start:start + max_batch]
            batch = torch.tensor([list(prompts[i]) for i in chunk],
                                 dtype=torch.long, device=device)
            for index, ids in zip(chunk,
                                  greedy_continue(model, batch,
                                                  max_new_tokens, eos_id)):
                results[index] = ids
    return results


def safe_extract(config: dict, text: str) -> str | None:
    """Apply the G8 rule, returning None when the model emitted no answer cue.

    An un-finetuned model has never seen this format and will usually produce
    no cue at all.  That is a wrong answer, not a crash, so the failure is
    reported as ``None`` and scored as incorrect.

    Args:
        config: The language config, for the answer cue.
        text: Decoded model output.

    Returns:
        The extracted label, or None if the rule does not apply.
    """
    try:
        return extract_answer(config, text)
    except ValueError:
        return None


@torch.no_grad()
def exact_match(model: torch.nn.Module, examples: Sequence, tokenizer,
                config: dict, eos_id: int, device: str,
                max_new_tokens: int = 160, max_batch: int = 64) -> dict:
    """Greedy-decode each example and score the extracted label exactly.

    Args:
        model: Model in eval mode.
        examples: :class:`data.sft_data.SFTExample` instances.
        tokenizer: Loaded SentencePiece processor.
        config: The language config, for the answer cue.
        eos_id: Stop token.
        device: Torch device.
        max_new_tokens: Cap on generated tokens.
        max_batch: Largest generation batch.

    Returns:
        Exact-match rate, counts, and the decoded predictions for inspection.
    """
    was_training = model.training
    model.eval()

    prompts = [e.input_ids[:e.n_prompt] for e in examples]
    generated = generate_completions(model, prompts, eos_id, device,
                                     max_new_tokens, max_batch)

    correct = 0
    no_cue = 0
    unterminated = 0
    predictions = []
    for example, ids in zip(examples, generated):
        text = tokenizer.decode(list(ids))
        predicted = safe_extract(config, text)
        if predicted is None:
            no_cue += 1
        if len(ids) >= max_new_tokens:
            unterminated += 1
        hit = predicted == example.answer
        correct += hit
        predictions.append({
            "id": example.record_id,
            "gold": example.answer,
            "predicted": predicted,
            "correct": bool(hit),
            "generated_tokens": len(ids),
            "text": text,
        })

    if was_training:
        model.train()

    total = len(examples)
    return {
        "exact_match": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "no_answer_cue": no_cue,
        "hit_length_cap": unterminated,
        "predictions": predictions,
    }
