"""Figures for the pretraining LR sweep (val loss vs LR, val curves per LR).

Pure plotting: reads the per-run summaries ``train.lr_sweep`` produced and
draws them, so figures can be regenerated without retraining.
"""

from __future__ import annotations

from pathlib import Path


def plot(results: list[dict], decision: dict, out_dir: Path,
         subtitle: str = "", stem_suffix: str = "") -> list[Path]:
    """Render the two sweep figures.

    Kept apart from the computation above so the numbers can be produced
    headless and the figures regenerated without retraining.

    Args:
        results: Per-run summaries.
        decision: Output of :func:`choose_winner`.
        out_dir: Where to write the figures.
        subtitle: Second title line naming the language, horizon and token
            budget.  Passed in rather than hardcoded because a figure that
            states a budget it did not run is worse than one with no subtitle.
        stem_suffix: Appended to both file names, so a variant's sweep does not
            overwrite the baseline's figures.

    Returns:
        Paths of the figures written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # -- validation curves, all four runs together --
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in sorted(results, key=lambda r: r["peak_lr"]):
        curve = result["val_curve"]
        if not curve:
            continue
        label = f"{result['peak_lr']:.0e}"
        if not result["stable"]:
            label += " (unstable)"
        ax.plot([p["step"] for p in curve], [p["val_loss"] for p in curve],
                marker="o", markersize=3,
                linestyle="--" if not result["stable"] else "-", label=label)

    ax.set_title("Learning-rate sweep: validation loss during training\n"
                 + subtitle)
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Validation loss (nats/token)")
    ax.legend(title="peak learning rate")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"sweep_val_curves{stem_suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # -- the money figure: final val loss against LR --
    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = sorted(results, key=lambda r: r["peak_lr"])
    lrs = [r["peak_lr"] for r in ordered]
    finals = [r["final_val_loss"] for r in ordered]

    ax.plot(lrs, finals, marker="o", color="#2b6cb0", label="final val loss")
    for result in ordered:
        if not result["stable"]:
            ax.scatter([result["peak_lr"]], [result["final_val_loss"]],
                       marker="x", s=120, color="#c53030", zorder=5,
                       label="unstable (discarded)")
    if decision.get("winner"):
        ax.axvline(decision["winner"], linestyle=":", color="#2f855a",
                   label=f"winner {decision['winner']:.0e}")

    ax.set_xscale("log")
    ax.set_title("Final validation loss vs peak learning rate\n" + subtitle)
    ax.set_xlabel("Peak learning rate (log scale)")
    ax.set_ylabel("Final validation loss (nats/token)")
    # De-duplicate the legend: the unstable marker may be added several times.
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"sweep_val_loss_vs_lr{stem_suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written
