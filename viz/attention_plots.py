"""Attention figures: heatmaps, per-head statistics, pre/post-finetune profiles.

Pure plotting.  The numbers come from ``eval.attention`` (pretraining text) and
``eval.attention_reasoning`` (reasoning prompts); nothing here runs a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_heatmaps(result: dict, out_path: Path, layers: tuple[int, ...],
                  heads: tuple[int, ...], seq_len: int = 48) -> Path:
    """Heatmaps for selected layers and heads.

    Args:
        result: Output of :func:`analyse`.
        out_path: Destination PNG.
        layers: Layer indices to show.
        heads: Head indices to show.
        seq_len: Positions to display, from the start of the example.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(layers), len(heads),
                             figsize=(3.1 * len(heads), 3.0 * len(layers)),
                             squeeze=False)
    image = None
    for row, layer in enumerate(layers):
        weights = result["_attention"][layer][0].cpu().numpy()
        for col, head in enumerate(heads):
            ax = axes[row][col]
            image = ax.imshow(weights[head, :seq_len, :seq_len],
                              cmap="viridis", aspect="auto",
                              vmin=0.0, interpolation="nearest")
            ax.set_title(f"Layer {layer}, head {head}", fontsize=10)
            if row == len(layers) - 1:
                ax.set_xlabel("Key position (attended to)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Query position", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Causal self-attention weights — {result['language'].capitalize()}\n"
        f"Lower-triangular structure is the causal mask: no query attends "
        f"to a later key",
        fontsize=12)
    fig.colorbar(image, ax=axes, shrink=0.6, label="Attention weight")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_statistics(result: dict, out_path: Path) -> Path:
    """Per-layer entropy and mean distance, one line per head.

    Args:
        result: Output of :func:`analyse`.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers, n_heads = result["n_layers"], result["n_heads"]
    entropy = np.array([l["entropy_normalised"] for l in result["layers"]])
    distance = np.array([l["mean_distance"] for l in result["layers"]])
    layers = np.arange(n_layers)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for head in range(n_heads):
        left.plot(layers, entropy[:, head], marker="o", markersize=4,
                  alpha=0.75, label=f"head {head}")
        right.plot(layers, distance[:, head], marker="o", markersize=4,
                   alpha=0.75, label=f"head {head}")

    left.plot(layers, entropy.mean(axis=1), color="black", linewidth=2.5,
              label="mean")
    right.plot(layers, distance.mean(axis=1), color="black", linewidth=2.5,
               label="mean")

    left.set_title("Attention entropy by layer and head\n"
                   "normalised: 0 = one key, 1 = uniform over visible keys")
    left.set_xlabel("Layer")
    left.set_ylabel("Normalised entropy")
    left.set_ylim(0, 1)
    left.grid(alpha=0.3)
    left.legend(fontsize=7, ncol=2)

    right.set_title("Mean attention distance by layer and head\n"
                    "attention-weighted (query − key), in token positions")
    right.set_xlabel("Layer")
    right.set_ylabel("Mean distance (positions)")
    right.grid(alpha=0.3)
    right.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{result['language'].capitalize()} — attention statistics "
                 f"over {result['n_examples']} held-out sequences", fontsize=13)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_reasoning_heatmaps(maps: dict, layers: tuple[int, int], head: int,
                  language: str, out_path: Path, seq_len: int = 48) -> Path:
    """Pretrained against finetuned, one early and one late layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0), squeeze=False)
    image = None
    for row, layer in enumerate(layers):
        for col, tag in enumerate(("pretrained", "finetuned")):
            ax = axes[row][col]
            weights = maps[tag][layer][head].numpy()
            image = ax.imshow(weights[:seq_len, :seq_len], cmap="viridis",
                              aspect="auto", vmin=0.0, interpolation="nearest")
            ax.set_title(f"{tag} — layer {layer}, head {head}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Key position (attended to)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Query position", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Attention on a comparative-reasoning prompt — {language.capitalize()}\n"
        f"same prompt, same token ids; the only difference is finetuning",
        fontsize=12)
    fig.colorbar(image, ax=axes, shrink=0.6, label="Attention weight")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_layer_profiles(before: dict, after: dict, language: str,
                        out_path: Path) -> Path:
    """Per-layer distance and entropy, pretrained against finetuned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(len(before["per_layer_distance"])))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(layers, before["per_layer_distance"], "o-", label="pretrained")
    axes[0].plot(layers, after["per_layer_distance"], "s-", label="finetuned")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean attention distance (positions)")
    axes[0].set_title("Local vs long-range")
    axes[0].legend(frameon=False)

    axes[1].plot(layers, before["per_layer_entropy"], "o-", label="pretrained")
    axes[1].plot(layers, after["per_layer_entropy"], "s-", label="finetuned")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Normalised entropy")
    axes[1].set_title("How spread out the heads are")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Attention profile before and after finetuning — "
                 f"{language.capitalize()}, comparative-reasoning prompts",
                 fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
