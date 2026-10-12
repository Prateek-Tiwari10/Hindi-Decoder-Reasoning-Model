"""Phase 3.2 -- attention analysis on the FINETUNED checkpoints.

WHAT THE BRIEF ASKS FOR
-----------------------
Reuse the Phase 2 attention toolkit on the finetuned checkpoints, compare
pretrained against finetuned heatmaps for at least one early and one late layer
per model, and say whether finetuning changed local versus long-range attention
or head specialisation -- especially on comparative-reasoning prompts.

WHY THE PROMPTS COME FROM THE TEST SLICES
------------------------------------------
Phase 2's ``eval.attention`` samples windows out of the pretraining corpus, which
is the right stimulus for a language model and the wrong one here: it would
measure how the two checkpoints read news text, not how they read a comparison.
The stimulus is therefore drawn from ``rel_id_d23`` -- the in-distribution
relation slice, on HELD-OUT entity names -- so both models see exactly the
prompts the reasoning task is scored on.

Both models see identical token ids. Any difference in the maps is the
finetuning, not the input.

WHAT IS MEASURED
----------------
Per layer and head, on the prompt tokens only:

  * normalised entropy -- how spread out a head's attention is, scaled by the
    number of keys each query may legally see, so early and late positions are
    comparable.
  * mean attention distance -- how far back, in positions, a head's weight sits.
    This is the local-versus-long-range axis the brief asks about.

Head specialisation is read off the spread of those per-head values within a
layer: a layer whose heads all look alike is not specialised, and a layer whose
heads separate is.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from eval.attention import collect_attention, head_statistics
from viz.attention_plots import plot_layer_profiles, plot_reasoning_heatmaps
from model.config import ModelConfig, Paths
from model import build_model

LOGGER = logging.getLogger("eval.attention_reasoning")

#: The slice the stimulus comes from: relation-only, trained depths, held-out
#: names.  Comparative reasoning with nothing else varying.
STIMULUS_SLICE = "rel_id_d23"

#: Enough prompts for the per-head means to be stable; attention is cheap.
N_PROMPTS = 64

#: Which finetuned size to compare against the base.  60k is the largest.
FT_SIZE = "60k"

LANG_CODE = {"hindi": "hi"}


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, ModelConfig]:
    """Build a model from a checkpoint without consulting the config on disk.

    ``eval.lm_metrics.load_trained`` cross-checks against ``<lang>/configs/``, which
    is right when scoring a run but wrong here: the point is to load two
    checkpoints that were written at different phases and compare them.  The
    architecture is taken from each checkpoint's own record.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(state["model_config"])
    model = build_model(config, device=device, arch=state.get("arch", "gpt"))
    model.load_state_dict(state["model"])
    model.eval()
    LOGGER.info("loaded %s (step %s, %s params)", path, state.get("step"),
                f"{config.parameter_count:,}")
    return model, config


def load_prompts(root: Path, language: str, n: int) -> list[str]:
    """Take the first ``n`` prompts from the stimulus slice.

    The slice is shuffled at generation time, so a prefix is a representative
    sample rather than a biased one.
    """
    code = LANG_CODE[language]
    path = (Paths.for_variant("hindi", root).finetune_data / "test" / code / "in_domain"
            / f"{STIMULUS_SLICE}.jsonl")
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= n:
                break
            prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts


def measure(model: torch.nn.Module, config: ModelConfig,
            token_lists: list[list[int]], device: str) -> dict:
    """Average per-head entropy and distance over the prompt set.

    Each prompt is run at its own true length rather than padded to a common
    one: padding would put mass on positions that are not really there and
    inflate the distance statistic.
    """
    layers = config.n_layers
    heads = config.n_heads
    entropy = [[0.0] * heads for _ in range(layers)]
    distance = [[0.0] * heads for _ in range(layers)]

    with torch.no_grad():
        for ids in token_lists:
            tokens = torch.tensor([ids], device=device)
            attentions = collect_attention(model, tokens)
            for layer, attention in enumerate(attentions):
                stats = head_statistics(attention)
                for head in range(heads):
                    entropy[layer][head] += stats["entropy_normalised"][head]
                    distance[layer][head] += stats["mean_distance"][head]

    n = len(token_lists)
    entropy = [[v / n for v in row] for row in entropy]
    distance = [[v / n for v in row] for row in distance]
    return {
        "per_head_entropy_normalised": entropy,
        "per_head_mean_distance": distance,
        "per_layer_entropy": [sum(r) / len(r) for r in entropy],
        "per_layer_distance": [sum(r) / len(r) for r in distance],
        # Spread of head values within a layer: the specialisation signal.
        "per_layer_head_distance_spread": [max(r) - min(r) for r in distance],
        "per_layer_head_entropy_spread": [max(r) - min(r) for r in entropy],
    }


def representative_maps(model: torch.nn.Module, ids: list[int],
                        device: str) -> list[torch.Tensor]:
    """Attention for a single prompt, kept on CPU for plotting."""
    with torch.no_grad():
        tokens = torch.tensor([ids], device=device)
        return [a[0].cpu() for a in collect_attention(model, tokens)]


def analyse_language(language: str, root: Path, device: str,
                     figures: Path, n_prompts: int = N_PROMPTS,
                     base_path: Path | None = None, ft_path: Path | None = None,
                     stem: str = "gpt") -> dict:
    """Compare pretrained vs finetuned attention for one model pair.

    Args:
        language: ``"hindi"``.
        root: Repository root.
        device: Torch device.
        figures: Output directory for the figures.
        n_prompts: Reasoning prompts to average over.
        base_path: Pretrained checkpoint; defaults to the GPT baseline.
        ft_path: Finetuned checkpoint; defaults to GPT-ft (``ft_60k``).
        stem: Tag used in figure file names (``gpt``, ``llama``).

    Returns:
        Per-layer / per-head statistics, deltas and figure paths.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, "16k")
    base_path = base_path or paths.final_model
    ft_path = ft_path or paths.gpt_finetuned
    for p in (base_path, ft_path):
        if not p.exists():
            raise SystemExit(f"missing checkpoint: {p}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
    prompts = load_prompts(root, language, n_prompts)
    token_lists = [tokenizer.encode(p)[:512] for p in prompts]
    LOGGER.info("%s: %d prompts, median length %d tokens", language,
                len(token_lists),
                sorted(len(t) for t in token_lists)[len(token_lists) // 2])

    results, maps = {}, {}
    for tag, path in (("pretrained", base_path), ("finetuned", ft_path)):
        model, config = load_checkpoint(path, device)
        results[tag] = measure(model, config, token_lists, device)
        maps[tag] = representative_maps(model, token_lists[0], device)
        results[tag]["checkpoint"] = str(path)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    n_layers = len(results["pretrained"]["per_layer_distance"])
    early, late = 0, n_layers - 1

    figure_paths = {
        "heatmaps": str(plot_reasoning_heatmaps(
            maps, (early, late), 0, language,
            figures / f"attention_ft_heatmaps_{stem}.png")),
        "profiles": str(plot_layer_profiles(
            results["pretrained"], results["finetuned"], language,
            figures / f"attention_ft_profile_{stem}.png")),
    }

    before, after = results["pretrained"], results["finetuned"]
    deltas = {
        "per_layer_distance_delta": [
            a - b for a, b in zip(after["per_layer_distance"],
                                  before["per_layer_distance"])],
        "per_layer_entropy_delta": [
            a - b for a, b in zip(after["per_layer_entropy"],
                                  before["per_layer_entropy"])],
        "head_distance_spread_delta": [
            a - b for a, b in zip(after["per_layer_head_distance_spread"],
                                  before["per_layer_head_distance_spread"])],
    }

    # The single head that moved furthest, named so the claim is checkable.
    moved = []
    for layer in range(n_layers):
        for head in range(len(before["per_head_mean_distance"][layer])):
            moved.append((
                after["per_head_mean_distance"][layer][head]
                - before["per_head_mean_distance"][layer][head], layer, head))
    moved.sort(key=lambda t: abs(t[0]), reverse=True)

    return {
        "language": language,
        "n_prompts": len(token_lists),
        "stimulus_slice": STIMULUS_SLICE,
        "finetuned_size": FT_SIZE,
        "early_layer": early,
        "late_layer": late,
        "pretrained": before,
        "finetuned": after,
        "deltas": deltas,
        "largest_head_shifts": [
            {"layer": l, "head": h, "distance_delta": round(d, 3)}
            for d, l, h in moved[:5]],
        "figures": figure_paths,
    }


def main() -> None:
    """Compare pretrained and finetuned attention on reasoning prompts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures = args.figures or root / "reports" / "gpt_finetune" / "figures"
    report = args.report or root / "reports" / "gpt_finetune" / "attention_finetuned.json"

    languages = ("hindi",)
    payload = {lang: analyse_language(lang, root, device, figures,
                                      args.n_prompts)
               for lang in languages}

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    for lang, result in payload.items():
        print(f"  {lang}: figures -> {', '.join(result['figures'].values())}")


if __name__ == "__main__":
    main()
