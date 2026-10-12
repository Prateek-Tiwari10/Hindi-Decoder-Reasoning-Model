"""Ablation -- does the no-positional model still know about order?

WHY THIS PROBE EXISTS
---------------------
Deleting the positional embedding table removes the only component that
*explicitly* tells the model where a token sits.  The tempting conclusion is
that the ablated model must therefore be order-blind, and that whatever
perplexity it still achieves comes from bag-of-words statistics alone.

That conclusion is wrong, and the reason is worth stating precisely.

Write ``A`` for one causal-attention layer with no positional input.  The
output at query position ``t`` is a function of the query vector at ``t`` and
the *multiset* of key/value vectors at positions ``0..t``.  Permuting the
earlier positions permutes the multiset's order but not the multiset, so a
**one-layer** no-positional causal model is exactly invariant to any
permutation of its prefix.  Its prediction at the last position cannot depend
on word order at all.

From **two layers** on this breaks.  The second layer's key at position ``j``
is the first layer's *output* at ``j``, which already depends on the multiset
``0..j`` -- a different quantity for each ``j``.  Position ``j`` is thereby
tagged by how much context it was allowed to see, and the causal mask has
smuggled order back in.  The model can, in principle, recover position from
prefix length alone.

This module measures whether the trained 11-layer ablation actually does that,
rather than assuming it either way.  Three measurements:

**Probe A -- prefix permutation.**  Hold the final token fixed, randomly
permute every token before it, and compare the next-token distribution at the
final position.  A permutation-invariant model gives an identical distribution
(KL = 0, top-1 agreement = 1.0).  Anything above zero is order sensitivity that
had to come from the causal mask, since there is no positional table.

**Probe B -- local swap.**  The same comparison but swapping a single adjacent
pair near the end of the prefix.  A full shuffle destroys the sentence; an
adjacent swap leaves the bag of words almost intact and isolates whether
*fine-grained* order matters to the prediction.

**Control -- an untrained one-layer no-positional model.**  Probe A run against
it must return ~0 (up to float error).  This is not a baseline to beat; it is a
check that the probe would in fact report invariance if invariance were there,
so a non-zero reading on the trained model means something.

Comparing all of this against the standard model, which has a positional table
and is therefore expected to be strongly order sensitive, gives the ablation's
cost in the currency that matters: not "did perplexity move" but "did the model
lose the ability to represent word order".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import ModelConfig, Paths
from data.token_stream import TokenStream
from eval.lm_metrics import load_trained
from model import build_model

LOGGER = logging.getLogger("ablation.order_probe")

# Fixed so the two models see byte-identical stimuli and byte-identical
# permutations.  A difference between the models is then a difference between
# the models, not between their draws.
PROBE_SEED = 20260916

N_SEQUENCES = 256
SEQ_LEN = 128
BATCH = 16


def sample_windows(stream: TokenStream, n: int, seq_len: int,
                   seed: int) -> np.ndarray:
    """Draw ``n`` non-overlapping-ish windows of ``seq_len`` tokens.

    Args:
        stream: The token stream to draw from.
        n: How many windows.
        seq_len: Window length.
        seed: RNG seed.

    Returns:
        ``(n, seq_len)`` int64 array.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(stream) - seq_len - 1, size=n)
    tokens = np.asarray(stream._tokens)          # memmap -> array view
    return np.stack([tokens[s:s + seq_len] for s in starts]).astype(np.int64)


def permute_prefix(batch: torch.Tensor, generator: torch.Generator
                   ) -> torch.Tensor:
    """Shuffle positions ``0..T-2``, leaving the final token in place.

    The final token is held fixed because it is the attention *query* for the
    prediction being compared.  Moving it would change the query as well as the
    keys, and the comparison would no longer isolate order.

    Args:
        batch: ``(B, T)`` token ids.
        generator: Torch RNG.

    Returns:
        ``(B, T)`` with each row's prefix independently permuted.
    """
    B, T = batch.shape
    out = batch.clone()
    for b in range(B):
        order = torch.randperm(T - 1, generator=generator, device=batch.device)
        out[b, :T - 1] = batch[b, order]
    return out


def swap_adjacent(batch: torch.Tensor, generator: torch.Generator,
                  window: int = 16) -> torch.Tensor:
    """Swap one adjacent pair drawn from the last ``window`` prefix positions.

    Restricted to the tail because attention weight concentrates near the
    query; a swap 100 tokens back would be a weaker perturbation for reasons of
    attention range rather than order encoding.

    Args:
        batch: ``(B, T)`` token ids.
        generator: Torch RNG.
        window: How far back from the query the swap may land.

    Returns:
        ``(B, T)`` with one adjacent transposition per row.
    """
    B, T = batch.shape
    out = batch.clone()
    lo = max(0, T - 1 - window)
    for b in range(B):
        i = int(torch.randint(lo, T - 2, (1,), generator=generator,
                              device=batch.device))
        out[b, i], out[b, i + 1] = batch[b, i + 1].clone(), batch[b, i].clone()
    return out


def final_distribution(model: torch.nn.Module, batch: torch.Tensor
                       ) -> torch.Tensor:
    """Next-token log-probabilities at the last position.

    Args:
        model: A decoder LM in eval mode.
        batch: ``(B, T)`` token ids.

    Returns:
        ``(B, V)`` log-probabilities.
    """
    with torch.no_grad():
        logits, _, _ = model(batch, last_position_only=True)
    return F.log_softmax(logits[:, -1, :].float(), dim=-1)


def compare(model: torch.nn.Module, original: torch.Tensor,
            perturbed: torch.Tensor) -> dict[str, float]:
    """Distance between predictions on original and perturbed inputs.

    Three numbers because they fail differently.  Symmetric KL is unbounded and
    dominated by the tails; total variation is bounded in [0, 1] and easy to
    read; top-1 agreement is what actually changes the generated text.

    Args:
        model: The model under test.
        original: ``(B, T)`` unperturbed ids.
        perturbed: ``(B, T)`` perturbed ids.

    Returns:
        Mean symmetric KL (nats), mean total variation, top-1 agreement rate.
    """
    log_p = final_distribution(model, original)
    log_q = final_distribution(model, perturbed)
    p, q = log_p.exp(), log_q.exp()

    kl_pq = (p * (log_p - log_q)).sum(-1)
    kl_qp = (q * (log_q - log_p)).sum(-1)
    sym_kl = 0.5 * (kl_pq + kl_qp)
    tv = 0.5 * (p - q).abs().sum(-1)
    agree = (log_p.argmax(-1) == log_q.argmax(-1)).float()

    return {
        "sym_kl_nats": float(sym_kl.mean()),
        "total_variation": float(tv.mean()),
        "top1_agreement": float(agree.mean()),
    }


def mean_of_batches(model: torch.nn.Module, windows: np.ndarray,
                    perturb, device: str, seed: int) -> dict[str, float]:
    """Run :func:`compare` over all windows in batches and average.

    Args:
        model: The model under test.
        windows: ``(N, T)`` token ids.
        perturb: Callable ``(batch, generator) -> batch``.
        device: Torch device.
        seed: RNG seed, re-set per model so both models see the same
            perturbations.

    Returns:
        Averaged metrics.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, float] = {}
    n_batches = 0
    for start in range(0, len(windows), BATCH):
        chunk = torch.from_numpy(windows[start:start + BATCH]).to(device)
        stats = compare(model, chunk, perturb(chunk, generator))
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    return {key: value / n_batches for key, value in totals.items()}


def invariance_control(config: ModelConfig, windows: np.ndarray,
                       device: str) -> dict[str, float]:
    """Probe A against an untrained *one-layer* no-positional model.

    One layer is the case the argument in this module's docstring says must be
    exactly permutation invariant.  If the probe reports a non-zero distance
    here, the probe is broken and every other number in this file is suspect.

    Args:
        config: The ablated model's config, copied down to one layer.
        windows: ``(N, T)`` token ids.
        device: Torch device.

    Returns:
        Probe A metrics; expected KL ~0 and agreement ~1.
    """
    spec = config.to_dict()
    spec["n_layers"] = 1
    one_layer = build_model(ModelConfig.from_dict(spec), device=device,
                            verify_budget=False)
    one_layer.eval()
    return mean_of_batches(one_layer, windows, permute_prefix, device,
                           PROBE_SEED)


def run(root: Path, device: str, report: Path | None) -> dict:
    """Run every probe against both models and the control.

    Args:
        root: Repository root.
        device: Torch device.
        report: Where to write JSON, or None.

    Returns:
        The results dictionary.
    """
    models = {
        "standard": ("16k", Paths.for_variant("hindi", root, "16k").final_model),
        "ablated": ("16k_nopos", Paths.for_variant("hindi", root, "16k_nopos").final_model),
    }

    paths = Paths.for_variant("hindi", root, "16k")
    stream = TokenStream(paths.split_bin("test"), SEQ_LEN)
    windows = sample_windows(stream, N_SEQUENCES, SEQ_LEN, PROBE_SEED)
    LOGGER.info("stimulus: %d windows of %d tokens from %s",
                len(windows), SEQ_LEN, paths.split_bin("test").name)

    results: dict = {
        "seed": PROBE_SEED,
        "n_sequences": int(len(windows)),
        "seq_len": SEQ_LEN,
        "models": {},
    }

    ablated_config = None
    for name, (variant, checkpoint) in models.items():
        model, config = load_trained("hindi", checkpoint, device, root, variant)
        if name == "ablated":
            ablated_config = config
        entry = {
            "variant": variant,
            "positional": config.positional,
            "parameters": int(config.parameter_count),
            "prefix_permutation": mean_of_batches(
                model, windows, permute_prefix, device, PROBE_SEED),
            "adjacent_swap": mean_of_batches(
                model, windows, swap_adjacent, device, PROBE_SEED + 1),
        }
        results["models"][name] = entry
        LOGGER.info("%-9s permute: KL %.4f TV %.4f top1 %.3f | swap: KL %.4f "
                    "TV %.4f top1 %.3f", name,
                    entry["prefix_permutation"]["sym_kl_nats"],
                    entry["prefix_permutation"]["total_variation"],
                    entry["prefix_permutation"]["top1_agreement"],
                    entry["adjacent_swap"]["sym_kl_nats"],
                    entry["adjacent_swap"]["total_variation"],
                    entry["adjacent_swap"]["top1_agreement"])
        del model
        torch.cuda.empty_cache()

    LOGGER.info("control: untrained 1-layer no-positional model ...")
    results["control_one_layer_nopos"] = invariance_control(
        ablated_config, windows, device)
    LOGGER.info("control  permute: KL %.2e TV %.2e top1 %.3f",
                results["control_one_layer_nopos"]["sym_kl_nats"],
                results["control_one_layer_nopos"]["total_variation"],
                results["control_one_layer_nopos"]["top1_agreement"])

    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("report -> %s", report)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/ablation/order_probe.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    run(args.root, args.device, args.report)


if __name__ == "__main__":
    main()
