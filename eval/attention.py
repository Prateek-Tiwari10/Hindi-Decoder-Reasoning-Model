"""Part 7 -- what the hand-built attention actually learned.

THE TWO STATISTICS, DEFINED
---------------------------
Both are computed per (layer, head) and averaged over query positions and
examples.  Let ``a[i, j]`` be the attention weight from query position ``i`` to
key position ``j``, so ``sum_j a[i, j] = 1`` and ``a[i, j] = 0`` for ``j > i``.

**Attention entropy**::

    H(i) = - sum_j a[i, j] * log(a[i, j])

Measured in nats.  It answers "how many keys is this query effectively looking
at?".  ``H = 0`` means the head puts all its weight on exactly one position --
a hard pointer.  The maximum for a query at position ``i`` is ``log(i + 1)``,
attained when the head spreads weight uniformly over everything it may see, i.e.
it is averaging rather than selecting.

Because that ceiling grows with position, raw entropy is not comparable across
positions.  :func:`head_statistics` therefore also reports *normalised* entropy,
``H(i) / log(i + 1)``, which lives in [0, 1] for every query and can be averaged
honestly.

**Mean attention distance**::

    D(i) = sum_j a[i, j] * (i - j)

The attention-weighted average of how far back a query looks, in token
positions.  ``i - j`` rather than ``|i - j|`` because the causal mask already
guarantees ``j <= i``; the distance is always backwards, and using the absolute
value would only obscure that.  ``D = 0`` means the head attends to the current
token; large ``D`` means it reaches back across the context.

WHAT TO LOOK FOR
----------------
The expected pattern in a trained decoder is that early layers are more local
and more peaked -- assembling characters and morphemes into words, which needs
only a few neighbouring positions -- while later layers are broader and reach
further back, doing the topical and syntactic work that needs the whole context.
If that pattern does *not* appear it is worth saying so rather than asserting
the textbook story.

Position 0 is excluded from both statistics.  It has exactly one visible key
(itself), so its attention is forced to 1.0 by the mask rather than learned: its
entropy is identically 0 and its distance identically 0, and including it would
drag every head's average towards zero for a reason that has nothing to do with
what the head learned.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from model.config import Paths
from data.token_stream import TokenStream
from eval.lm_metrics import load_trained
from viz.attention_plots import plot_heatmaps, plot_statistics

LOGGER = logging.getLogger("attention")

#: Below this, a weight contributes ~0 to the entropy sum but its log is -inf.
EPSILON = 1e-12


@torch.no_grad()
def collect_attention(model: torch.nn.Module, tokens: torch.Tensor
                      ) -> list[torch.Tensor]:
    """Run the model and return every layer's attention weights.

    Args:
        model: Trained model in eval mode.  Must be in eval mode, or attention
            dropout would randomly zero weights and the statistics would
            measure the dropout mask rather than the head.
        tokens: Token ids, ``(B, T)``.

    Returns:
        ``n_layers`` tensors of shape ``(B, H, T, T)``.
    """
    _, _, attentions = model(tokens, return_attention=True)
    return attentions


def head_statistics(attention: torch.Tensor) -> dict:
    """Entropy and mean distance for every head of one layer.

    Args:
        attention: ``(B, H, T, T)`` post-softmax weights.

    Returns:
        Per-head lists of entropy (nats), normalised entropy, and mean
        attention distance (positions).
    """
    B, H, T, _ = attention.shape
    a = attention.double()

    # Drop query position 0: its distribution is forced by the mask, not learned.
    a = a[:, :, 1:, :]
    queries = torch.arange(1, T, device=a.device, dtype=torch.double)

    entropy = -(a * torch.log(a + EPSILON)).sum(dim=-1)      # (B, H, T-1)
    # Ceiling for query i is log(i + 1): the number of keys it may attend to.
    ceiling = torch.log(queries + 1.0).view(1, 1, -1)
    normalised = entropy / ceiling

    keys = torch.arange(T, device=a.device, dtype=torch.double).view(1, 1, 1, -1)
    # i - j, non-negative everywhere the mask allows weight.
    offsets = queries.view(1, 1, -1, 1) - keys
    distance = (a * offsets).sum(dim=-1)                     # (B, H, T-1)

    return {
        "entropy_nats": entropy.mean(dim=(0, 2)).tolist(),
        "entropy_normalised": normalised.mean(dim=(0, 2)).tolist(),
        "mean_distance": distance.mean(dim=(0, 2)).tolist(),
    }


def analyse(language: str, device: str, n_examples: int, seq_len: int,
            seed: int = 0, root: Path = Path("."), variant: str = "16k",
            checkpoint: Path | None = None) -> dict:
    """Compute per-layer, per-head statistics for one model.

    Args:
        language: ``"hindi"`` (the only language).
        device: Torch device.
        n_examples: Sequences to average over.
        seq_len: Length of each sequence.
        seed: RNG seed for sampling sequences.
        root: Repository root.
        variant: Architecture variant, selecting the checkpoint and bins.
        checkpoint: Override the checkpoint; defaults to the variant's
            ``final.pt``.

    Returns:
        Statistics plus the example used for the heatmaps.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, variant)
    model, config = load_trained(language, checkpoint or paths.final_model,
                                 device, root, variant)
    tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
    stream = TokenStream(paths.split_bin("test"), config.context_len)

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(stream) - seq_len - 1, size=n_examples)
    batch = np.stack([np.asarray(stream.tokens[s:s + seq_len], dtype=np.int64)
                      for s in starts])
    tokens = torch.from_numpy(batch).to(device)

    attentions = collect_attention(model, tokens)
    layers = [head_statistics(a.cpu()) for a in attentions]

    return {
        "language": language,
        "n_examples": n_examples,
        "seq_len": seq_len,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "layers": layers,
        "example_text": tokenizer.decode(batch[0].tolist())[:300],
        "_attention": attentions,
        "_tokens": batch[0],
        "_tokenizer": tokenizer,
    }


def verify_causal(result: dict) -> dict:
    """Confirm every attention map is lower-triangular.

    A full-stack correctness check: if any weight sits above the diagonal in a
    trained model, the mask leaked and every number in Parts 4-6 is suspect.

    Args:
        result: Output of :func:`analyse`.

    Returns:
        The largest weight found strictly above the diagonal.
    """
    worst = 0.0
    for attention in result["_attention"]:
        T = attention.shape[-1]
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        worst = max(worst, attention[:, :, upper].abs().max().item())
    return {"max_weight_above_diagonal": worst, "lower_triangular": worst == 0.0}


def uniform_baseline(seq_len: int) -> float:
    """Mean attention distance of a head that attends uniformly.

    The reference point every measured distance should be read against. A
    causal head spreading its weight evenly over the ``i + 1`` keys it may see
    has expected distance ``i / 2`` at query ``i``; averaged over queries
    ``1..T-1`` that is close to ``T / 4``... but the value actually wanted here
    is the average of ``i / 2``, which is what this computes. A layer sitting at
    this number is averaging over its whole visible context rather than
    selecting anything.

    Args:
        seq_len: Sequence length the statistics were measured over.

    Returns:
        Mean distance of a uniform causal head.
    """
    if seq_len < 2:
        return 0.0
    return sum(i / 2 for i in range(1, seq_len)) / (seq_len - 1)


def summarise(result: dict) -> dict:
    """Reduce to the numbers the report quotes.

    Args:
        result: Output of :func:`analyse`.

    Returns:
        Per-layer means and the early-vs-late comparison.
    """
    entropy = np.array([l["entropy_normalised"] for l in result["layers"]])
    distance = np.array([l["mean_distance"] for l in result["layers"]])
    seq_len = result["seq_len"]

    return {
        "per_layer_mean_normalised_entropy": entropy.mean(axis=1).tolist(),
        "per_layer_mean_distance": distance.mean(axis=1).tolist(),
        "first_layer_mean_distance": float(distance[0].mean()),
        "last_layer_mean_distance": float(distance[-1].mean()),
        # Comparing only first against last would call this network
        # "early layers more local" -- true by 31.5 vs 34.4, and badly
        # misleading, because the minimum is in the MIDDLE. The per-layer
        # profile is broad -> narrow -> broad, so the informative numbers are
        # where the narrowest layer actually is and how deep the trough goes.
        "narrowest_layer": int(distance.mean(axis=1).argmin()),
        "narrowest_layer_mean_distance": float(distance.mean(axis=1).min()),
        "broadest_layer": int(distance.mean(axis=1).argmax()),
        "broadest_layer_mean_distance": float(distance.mean(axis=1).max()),
        # A causal head attending uniformly over everything it may see scores
        # (T-1)/2 ~ T/2. Layer 0 sitting at that value means it is averaging,
        # not selecting -- which is the opposite of the usual "early = local"
        # story and needs to be reported as measured.
        "uniform_baseline_distance": uniform_baseline(seq_len),
        "most_local_head": {
            "layer": int(np.unravel_index(distance.argmin(), distance.shape)[0]),
            "head": int(np.unravel_index(distance.argmin(), distance.shape)[1]),
            "mean_distance": float(distance.min())},
        "most_global_head": {
            "layer": int(np.unravel_index(distance.argmax(), distance.shape)[0]),
            "head": int(np.unravel_index(distance.argmax(), distance.shape)[1]),
            "mean_distance": float(distance.max())},
        "most_peaked_head": {
            "layer": int(np.unravel_index(entropy.argmin(), entropy.shape)[0]),
            "head": int(np.unravel_index(entropy.argmin(), entropy.shape)[1]),
            "normalised_entropy": float(entropy.min())},
    }


def main() -> None:
    """Run the attention analysis for both models."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-examples", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained checkpoint to load. Defaults to the variant's final.pt. Needed because a reduced-budget run produces last.pt / best.pt rather than a promoted final.")
    parser.add_argument("--figures", type=Path, default=Path("reports/gpt_pretrain/figures"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/attention.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    languages = ("hindi",)

    reports = {}
    for language in languages:
        LOGGER.info("=== %s ===", language)
        result = analyse(language, device, args.n_examples, args.seq_len,
                         args.seed, args.root, args.variant,
                         args.checkpoint)

        causal = verify_causal(result)
        LOGGER.info("  causal check: max weight above diagonal = %.3e (%s)",
                    causal["max_weight_above_diagonal"],
                    "PASS" if causal["lower_triangular"] else "FAIL")

        # Early / mid / late, computed from the depth actually trained: the
        # baseline has 7 layers and Model3 has 11, so a hardcoded (0, 3, 6)
        # would silently stop sampling the late layers of the deeper stack --
        # the layers the width-vs-depth question is about.
        n_layers, n_heads = result["n_layers"], result["n_heads"]
        layers = tuple(sorted({0, n_layers // 2, n_layers - 1}))
        heads = tuple(range(0, n_heads, max(1, n_heads // 4)))[:4]
        suffix = Paths.for_variant(language, args.root, args.variant).suffix
        LOGGER.info("  heatmaps: layers %s of %d, heads %s of %d",
                    layers, n_layers, heads, n_heads)

        figures = [
            str(plot_heatmaps(
                result,
                args.figures / f"attention_heatmaps_{language}{suffix}.png",
                layers=layers, heads=heads)),
            str(plot_statistics(
                result,
                args.figures / f"attention_stats_{language}{suffix}.png")),
        ]

        summary = summarise(result)
        LOGGER.info("  mean distance by layer: %s",
                    [round(x, 1) for x in summary["per_layer_mean_distance"]])
        LOGGER.info("  narrowest layer %d (%.2f) vs uniform baseline %.2f; "
                    "broadest layer %d (%.2f)",
                    summary["narrowest_layer"],
                    summary["narrowest_layer_mean_distance"],
                    summary["uniform_baseline_distance"],
                    summary["broadest_layer"],
                    summary["broadest_layer_mean_distance"])

        reports[language] = {
            k: v for k, v in result.items() if not k.startswith("_")}
        reports[language]["causal_check"] = causal
        reports[language]["summary"] = summary
        reports[language]["figures"] = figures

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
