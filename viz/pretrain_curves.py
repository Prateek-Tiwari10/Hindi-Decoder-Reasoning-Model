"""Training-curve figures for Part 4.

Kept apart from ``lm/train.py`` deliberately.  The brief requires plotting
functions to be separate from computation functions, and there is a practical
reason too: the figures get regenerated many times while a report is being
written, and none of those regenerations should need a GPU or a checkpoint.
Everything here reads ``log.jsonl`` and nothing else.

READING THE FIGURES
-------------------
Train loss is the running mean over the logging interval, so it is noisy;
validation loss is a fixed number of held-out batches, so it is comparable
across steps and across runs.  The gap between them is the thing to watch: for
a single-epoch run over 500M tokens it should stay small, because the model
sees almost every token exactly once and has little opportunity to memorise.
A widening gap would be the first sign that the run has started overfitting,
which at this data-to-parameter ratio would itself be surprising.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Both curves use nats/token, matching the training objective.  Perplexity is
#: exp() of this and is reported separately in Part 5; plotting loss rather than
#: perplexity keeps the vertical scale readable, since exp() of an early loss
#: near 10 is ~22,000 and would flatten everything that follows.
Y_LABEL = "Loss (nats/token)"


def read_log(log_path: Path) -> list[dict]:
    """Load a training log.

    Args:
        log_path: Path to ``log.jsonl``.

    Returns:
        One record per logged step, in order.
    """
    return [json.loads(line)
            for line in Path(log_path).read_text().splitlines() if line.strip()]


def loss_curves(history: list[dict], language: str, out_path: Path,
                peak_lr: float | None = None,
                horizon: str = "one epoch over the training corpus") -> Path:
    """Plot train and validation loss against step.

    Args:
        history: Records from :func:`read_log`.
        language: Language name, for the title.
        out_path: Destination PNG.
        peak_lr: Peak learning rate, named in the subtitle if given.
        horizon: How much data the run saw.  Passed in because the reduced
            comparison runs are not one epoch, and a subtitle claiming a budget
            the run did not have is worse than no subtitle.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(r["step"], r["train_loss"]) for r in history if "train_loss" in r]
    val = [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if train:
        ax.plot([s for s, _ in train], [v for _, v in train],
                color="#90b8e0", linewidth=0.9, label="Train loss (running mean)")
    if val:
        ax.plot([s for s, _ in val], [v for _, v in val],
                color="#c53030", marker="o", markersize=3, linewidth=1.6,
                label="Validation loss (held-out)")

    subtitle = f"{language.capitalize()} — {horizon}"
    if peak_lr is not None:
        subtitle += f", peak LR {peak_lr:.0e}"
    ax.set_title(f"Pretraining loss curves\n{subtitle}")
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel(Y_LABEL)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def lr_and_grad(history: list[dict], language: str, out_path: Path) -> Path:
    """Plot the learning-rate schedule and gradient norm on twin axes.

    Two diagnostics that answer different questions than the loss does. The LR
    trace confirms the schedule actually ran as configured -- warmup reached the
    peak, cosine decayed to the floor -- rather than being asserted. The
    gradient norm shows whether clipping was active: a trace pinned at the clip
    threshold means nearly every step was clipped, which silently changes the
    effective learning rate from the one that was swept.

    Args:
        history: Records from :func:`read_log`.
        language: Language name, for the title.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in history if "lr" in r]
    lrs = [r["lr"] for r in history if "lr" in r]
    grads = [(r["step"], r["grad_norm"]) for r in history if "grad_norm" in r]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, lrs, color="#2f855a", linewidth=1.4,
            label="Learning rate")
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Learning rate", color="#2f855a")
    ax.tick_params(axis="y", labelcolor="#2f855a")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    if grads:
        twin.plot([s for s, _ in grads], [g for _, g in grads],
                  color="#b7791f", linewidth=0.8, alpha=0.75,
                  label="Gradient norm (pre-clip)")
    twin.axhline(1.0, color="#9b2c2c", linestyle=":", linewidth=1.2,
                 label="Clip threshold (1.0)")
    twin.set_ylabel("Gradient norm", color="#b7791f")
    twin.tick_params(axis="y", labelcolor="#b7791f")

    ax.set_title(f"Learning-rate schedule and gradient norm\n"
                 f"{language.capitalize()} — warmup then cosine to a 10% floor")

    handles = ax.get_legend_handles_labels()
    twin_handles = twin.get_legend_handles_labels()
    ax.legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1],
              loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def summarise(history: list[dict]) -> dict:
    """Reduce a training log to the numbers the report quotes.

    Args:
        history: Records from :func:`read_log`.

    Returns:
        Final and best losses, step count and clipping rate.
    """
    val = [r["val_loss"] for r in history if "val_loss" in r]
    train = [r["train_loss"] for r in history if "train_loss" in r]
    grads = [r["grad_norm"] for r in history if "grad_norm" in r]

    # Separate the two things a large grad_norm can mean.  Under fp16 the
    # GradScaler deliberately produces inf when a gradient overflows, then
    # SKIPS that step and halves the loss scale -- the mechanism working, not a
    # problem.  Lumping those in with genuine clipping (as an earlier version
    # did) both overstates the clipping rate and reports max_grad_norm as inf,
    # hiding the real maximum.
    finite = [g for g in grads if math.isfinite(g)]
    non_finite = [g for g in grads if not math.isfinite(g)]

    return {
        "steps_logged": len(history),
        "final_step": history[-1]["step"] if history else 0,
        "final_train_loss": train[-1] if train else None,
        "final_val_loss": val[-1] if val else None,
        "best_val_loss": min(val) if val else None,
        # A high clipping rate means the swept LR is not the LR that ran.
        "fraction_steps_clipped": (
            sum(g > 1.0 for g in finite) / len(finite) if finite else None),
        "max_grad_norm": max(finite) if finite else None,
        # Steps the GradScaler skipped on an fp16 overflow.  A handful is
        # normal; a large fraction would mean the loss scale never settled.
        "steps_skipped_fp16_overflow": len(non_finite),
        "fraction_steps_skipped": (
            len(non_finite) / len(grads) if grads else None),
    }


def render(language: str, log_path: Path, out_dir: Path,
           peak_lr: float | None = None, tag: str = "",
           horizon: str = "one epoch over the training corpus") -> dict:
    """Render both figures for one run and summarise it.

    Args:
        language: Language name.
        log_path: Path to ``log.jsonl``.
        out_dir: Directory for the figures.
        peak_lr: Peak learning rate, for the loss-curve subtitle.
        tag: Appended to both file names so a variant's curves do not overwrite
            the baseline's.
        horizon: How much data the run saw, for the subtitle.

    Returns:
        The summary, with the figure paths added.
    """
    history = read_log(log_path)
    summary = summarise(history)
    summary["figures"] = [
        str(loss_curves(history, language,
                        out_dir / f"loss_curves_{language}{tag}.png", peak_lr,
                        horizon)),
        str(lr_and_grad(history, language,
                        out_dir / f"lr_grad_{language}{tag}.png")),
    ]
    return summary


def main() -> None:
    """Render training curves from a log."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/gpt_pretrain/figures"))
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--tag", default="",
                        help='Suffix for the figure names, e.g. "_16k"')
    parser.add_argument("--horizon", default="one epoch over the training corpus",
                        help="How much data the run saw, for the subtitle")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    summary = render(args.language, args.log, args.out, args.peak_lr,
                     args.tag, args.horizon)
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
