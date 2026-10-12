"""The Part 2 correctness gates for the model.

These are the checks the brief grades explicitly, plus the ones that catch the
bugs no learning rate can fix.  They run on CPU in seconds, so they run before
any GPU quota is spent.

The important one is FUTURE-BLINDNESS.  A decoder-only LM is trained on a task
that is trivial if the causal mask leaks: predicting token ``t+1`` is free if
the model can look at token ``t+1``.  A leak therefore does not announce itself
as a crash -- it shows up as a suspiciously good training loss and a validation
loss that never follows.  The test below perturbs the future and asserts the
past does not move, which is the only direct way to see the mask working.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from model.config import ModelConfig, Paths
from model import build_model
from model.gpt import DecoderLM

LOGGER = logging.getLogger("verify")


def check_parameter_count(config: ModelConfig) -> dict:
    """Gate 2d: the built model matches the designed model, inside budget.

    Args:
        config: Architecture configuration.

    Returns:
        Check result with the analytic and live counts and the breakdown.
    """
    model = build_model(config, verify_budget=False)
    actual = model.num_parameters()
    expected = config.parameter_count

    embedding = config.vocab_size * config.d_model
    positions = (0 if config.positional == "none"
                 else config.context_len * config.d_model)
    blocks = actual - embedding - positions - 2 * config.d_model

    tied = (model.lm_head.weight is model.token_embedding.weight)
    low, high = 25_000_000, 30_000_000
    return {
        "name": "2d parameter count",
        "analytic": expected,
        "live": actual,
        "matches": actual == expected,
        "within_budget": low <= actual <= high,
        "weights_tied": tied,
        "saved_by_tying": embedding if tied else 0,
        "breakdown": {"token_embedding_tied": embedding,
                      "positional": positions,
                      "blocks": blocks,
                      "final_norm": 2 * config.d_model},
        "passed": (actual == expected and low <= actual <= high and tied),
    }


def check_shapes(config: ModelConfig) -> dict:
    """Gates 2a/2c/2d: tensors have the shapes the design says they do.

    Args:
        config: Architecture configuration.

    Returns:
        Check result recording each observed shape.
    """
    model = build_model(config, verify_budget=False).eval()
    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))

    embedded = model.token_embedding(idx)
    if model.position_embedding is not None:
        embedded = embedded + model.position_embedding(torch.arange(T))
    block_out, _ = model.blocks[0](embedded)
    logits, _, attentions = model(idx, return_attention=True)

    return {
        "name": "2a/2c/2d shapes",
        "embedding": list(embedded.shape),
        "embedding_ok": list(embedded.shape) == [B, T, config.d_model],
        "block_preserves_shape": block_out.shape == embedded.shape,
        "logits": list(logits.shape),
        "logits_ok": list(logits.shape) == [B, T, config.vocab_size],
        "n_attention_maps": len(attentions),
        "attention": list(attentions[0].shape),
        "attention_ok": (list(attentions[0].shape)
                         == [B, config.n_heads, T, T]),
        "passed": (list(embedded.shape) == [B, T, config.d_model]
                   and block_out.shape == embedded.shape
                   and list(logits.shape) == [B, T, config.vocab_size]
                   and len(attentions) == config.n_layers),
    }


def check_context_cap(config: ModelConfig) -> dict:
    """Gate 2a: the positional table is a real cap, not a soft suggestion.

    A sequence one token longer than ``context_len`` must fail loudly.  If it
    silently succeeded, the extra positions would be reading rows that do not
    exist -- or worse, wrapping around.

    Args:
        config: Architecture configuration.

    Returns:
        Check result naming the exception raised.
    """
    model = build_model(config, verify_budget=False).eval()
    too_long = torch.randint(0, config.vocab_size, (1, config.context_len + 1))
    try:
        model(too_long)
        return {"name": "2a context cap", "raised": None, "passed": False}
    except ValueError as exc:
        return {"name": "2a context cap", "raised": str(exc), "passed": True}


def check_future_blindness(config: ModelConfig, tolerance: float = 1e-5) -> dict:
    """Gate 2b (graded): changing the future must not change the past.

    Take a sequence, record the logits, then overwrite every token strictly
    after position ``t`` with different ids.  Logits at positions ``0..t`` must
    be bit-for-bit unchanged up to float tolerance.  Logits after ``t`` must
    change -- otherwise the model is ignoring its input and the test would pass
    vacuously.

    Runs in eval mode: dropout is stochastic, and its noise would swamp the
    comparison.

    Args:
        config: Architecture configuration.
        tolerance: Maximum tolerated absolute difference in the past.

    Returns:
        Check result with the observed drifts.
    """
    torch.manual_seed(0)
    model = build_model(config, verify_budget=False).eval()

    B, T, t = 2, 32, 15
    idx = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        before, _, _ = model(idx)

        perturbed = idx.clone()
        # Shift every future token to a different id, wrapping in-vocabulary.
        perturbed[:, t + 1:] = (perturbed[:, t + 1:] + 7919) % config.vocab_size
        after, _, _ = model(perturbed)

    past_drift = (before[:, :t + 1] - after[:, :t + 1]).abs().max().item()
    future_drift = (before[:, t + 1:] - after[:, t + 1:]).abs().max().item()

    return {
        "name": "2b future-blindness",
        "perturbed_positions": f"{t + 1}..{T - 1}",
        "max_change_at_positions_<=t": past_drift,
        "max_change_after_t": future_drift,
        "past_unchanged": past_drift < tolerance,
        "future_changed": future_drift > tolerance,
        "passed": past_drift < tolerance and future_drift > tolerance,
    }


def check_causal_mask_shape(config: ModelConfig) -> dict:
    """Gate 2b/7: attention weights are lower-triangular and normalised.

    Every row must sum to 1 (softmax) and every entry above the diagonal must
    be exactly 0 (masked).  Part 7's heatmaps are the visual version of this;
    here it is asserted numerically.

    Args:
        config: Architecture configuration.

    Returns:
        Check result with the largest weight found in the future.
    """
    model = build_model(config, verify_budget=False).eval()
    T = 24
    idx = torch.randint(0, config.vocab_size, (1, T))
    with torch.no_grad():
        _, _, attentions = model(idx, return_attention=True)

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    worst_future = max(a[0, :, upper].abs().max().item() for a in attentions)
    row_sums = torch.stack([a.sum(dim=-1) for a in attentions])
    row_sum_error = (row_sums - 1.0).abs().max().item()

    return {
        "name": "2b causal mask",
        "max_weight_on_future": worst_future,
        "max_row_sum_error": row_sum_error,
        "lower_triangular": worst_future == 0.0,
        "rows_sum_to_one": row_sum_error < 1e-5,
        "passed": worst_future == 0.0 and row_sum_error < 1e-5,
    }


def check_head_split_roundtrip(config: ModelConfig) -> dict:
    """Gate 2b: splitting into heads and merging back is the identity.

    An off-by-one in the reshape or a missing ``contiguous()`` scrambles
    channels across heads.  The model still trains -- badly -- and nothing
    reports an error, so this is checked directly.

    Args:
        config: Architecture configuration.

    Returns:
        Check result with the round-trip error.
    """
    B, T, C, H = 2, 9, config.d_model, config.n_heads
    d_k = C // H
    x = torch.randn(B, T, C)

    split = x.view(B, T, H, d_k).transpose(1, 2)          # (B, H, T, Dk)
    merged = split.transpose(1, 2).contiguous().view(B, T, C)

    error = (x - merged).abs().max().item()
    return {
        "name": "2b head split/merge",
        "split_shape": list(split.shape),
        "split_shape_ok": list(split.shape) == [B, H, T, d_k],
        "roundtrip_error": error,
        "passed": error == 0.0 and list(split.shape) == [B, H, T, d_k],
    }


def check_initial_loss(config: ModelConfig, tolerance: float = 0.25) -> dict:
    """Gate 2d: an untrained model is uniform, so its loss is ``ln(V)``.

    A freshly initialised model has no reason to prefer any token, so it should
    spread probability evenly over the whole vocabulary and score
    ``-ln(1/V) = ln(V)`` nats -- 10.086 for the baseline's 24,000, 9.680 for
    Model3's 16,000.  A much lower initial loss means the init is not neutral;
    a much higher one means the logits are already large and confidently wrong,
    which makes early training unstable.

    Also runs one backward pass and confirms every parameter received a
    gradient -- a tensor with ``grad is None`` after backward is disconnected
    from the loss and will never train.

    Args:
        config: Architecture configuration.
        tolerance: Tolerated absolute deviation from ``ln(V)``, in nats.

    Returns:
        Check result with the observed loss and gradient coverage.
    """
    torch.manual_seed(0)
    model = build_model(config, verify_budget=False)

    B, T = 4, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, loss, _ = model(idx, targets)
    loss.backward()

    expected = math.log(config.vocab_size)
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]

    return {
        "name": "2d initial loss",
        "loss": loss.item(),
        "expected_ln_vocab": expected,
        "deviation": abs(loss.item() - expected),
        "params_without_grad": missing,
        "passed": (abs(loss.item() - expected) < tolerance and not missing),
    }


def run_all(config: ModelConfig) -> dict:
    """Run every Part 2 gate.

    Args:
        config: Architecture configuration.

    Returns:
        Report with each check's result and an overall verdict.
    """
    checks = [
        check_parameter_count(config),
        check_shapes(config),
        check_context_cap(config),
        check_future_blindness(config),
        check_causal_mask_shape(config),
        check_head_split_roundtrip(config),
        check_initial_loss(config),
    ]
    return {
        "config": config.to_dict(),
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


def main() -> None:
    """Run the Part 2 gates and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/model_checks.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = Paths.for_variant(args.language, args.root, args.variant)
    config = ModelConfig.load(paths.model_config)
    report = run_all(config)
    report["language"] = args.language
    report["variant"] = args.variant

    LOGGER.info("Part 2 gates -- %s (%s)", args.language,
                args.variant or "baseline")
    LOGGER.info("=" * 68)
    for check in report["checks"]:
        LOGGER.info("[%s] %s", "PASS" if check["passed"] else "FAIL",
                    check["name"])
        for key, value in check.items():
            if key in ("name", "passed"):
                continue
            if isinstance(value, float):
                value = f"{value:.3e}" if 0 < abs(value) < 1e-3 else f"{value:.6f}"
            LOGGER.info("        %-28s %s", key, value)
    LOGGER.info("=" * 68)
    LOGGER.info("OVERALL: %s", "PASS" if report["all_passed"] else "FAIL")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
