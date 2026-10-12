"""Pieces shared by every decoder architecture in this repository.

Both the GPT-style and the LLaMA-style decoder use the same token embedding
convention, the same tied LM head, the same additive causal mask and the same
autoregressive sampler.  Keeping them here means the two architectures cannot
drift apart on anything except what the study deliberately varies.

Nothing in this module holds parameters of its own; it builds masks, counts
parameters and runs generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(size: int) -> torch.Tensor:
    """Additive causal mask of shape ``(1, 1, size, size)``.

    ``0.0`` where attention is allowed (key position <= query position) and
    ``-inf`` strictly above the diagonal.  Adding ``-inf`` to a score sends its
    softmax weight to exactly zero, and row ``t`` always allows column ``t``,
    so no row is ever fully masked and the softmax cannot produce NaN.

    Args:
        size: Sequence length the mask covers.

    Returns:
        Float tensor broadcastable over ``(B, H, T, T)`` score tensors.
    """
    mask = torch.full((size, size), float("-inf"))
    mask = torch.triu(mask, diagonal=1)          # strict upper triangle
    return mask.view(1, 1, size, size)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameter scalars, counting a tied (shared) tensor once.

    Args:
        model: Any module.
        trainable_only: Count only parameters requiring gradients.

    Returns:
        Number of scalars.
    """
    seen: dict[int, int] = {}
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        seen[id(param)] = param.numel()
    return sum(seen.values())


@torch.no_grad()
def generate(model: torch.nn.Module, prefix: torch.Tensor, max_new_tokens: int,
             temperature: float | None = None, eos_id: int | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Continue a batch of prefixes.

    Args:
        model: Trained model in eval mode.
        prefix: Prompt token ids, ``(B, T0)``.
        max_new_tokens: How many tokens to append.
        temperature: Sampling temperature; ``None`` means greedy (argmax).
        eos_id: Unused as a stop signal -- generation always runs the full
            length so every sample is the same length and the diversity
            statistics are comparable. Accepted for interface clarity.
        generator: Seeded RNG, so a reported sample can be reproduced.

    Returns:
        ``(B, T0 + max_new_tokens)`` including the prefix.
    """
    context_len = model.config.context_len
    tokens = prefix.clone()

    for _ in range(max_new_tokens):
        # The positional table has exactly context_len rows, so once the
        # sequence reaches that length the window must slide: keep the most
        # recent context_len tokens and drop the oldest.
        window = tokens[:, -context_len:]

        logits, _, _ = model(window, last_position_only=True)
        # Only the last position predicts the next token; the rest re-predict
        # tokens we already have.
        next_logits = logits[:, -1, :].float()

        if temperature is None:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1,
                                           generator=generator)

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens
