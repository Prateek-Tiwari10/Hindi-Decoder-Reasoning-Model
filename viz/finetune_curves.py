"""Figures for the finetuning runs.

    python -m viz.finetune_curves

Plotting is kept in functions separate from anything that computes -- an
assignment requirement, and the reason this module reads finished logs rather
than being called from inside the training loop.  Every figure carries a title,
axis labels and a legend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: One colour per architecture, held constant across every figure so a reader
#: does not have to re-check the legend on each plot.
COLOURS = {"gpt": "#1f77b4", "llama": "#d62728"}
LABELS = {"gpt": "GPT-style (finetuned)", "llama": "LLaMA-style (finetuned)"}


def read_log(path: Path) -> tuple[list[dict], list[dict]]:
    """Split a run log into its training records and its validation records.

    Args:
        path: ``log.jsonl`` from a run directory.

    Returns:
        ``(train_records, val_records)``.
    """
    train, val = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        (val if "val_loss" in record else train).append(record)
    return train, val


def smooth(values: list[float], window: int = 25) -> list[float]:
    """Trailing moving average, for readability only.

    The raw series is always plotted underneath at low opacity, so smoothing
    never hides what actually happened.

    Args:
        values: The series.
        window: Averaging window.

    Returns:
        The smoothed series, same length.
    """
    out = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_loss_curves(runs: dict[str, tuple[list[dict], list[dict]]],
                     out_path: Path) -> Path:
    """Train and validation loss against optimizer step, both languages.

    Args:
        runs: Language to ``(train_records, val_records)``.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for language, (train, val) in sorted(runs.items()):
        colour = COLOURS[language]
        steps = [r["step"] for r in train]
        losses = [r["train_loss"] for r in train]
        axis.plot(steps, losses, color=colour, alpha=0.18, linewidth=0.8)
        axis.plot(steps, smooth(losses), color=colour, linewidth=1.8,
                  label=f"{LABELS[language]} — train (smoothed)")
        if val:
            axis.plot([r["step"] for r in val], [r["val_loss"] for r in val],
                      color=colour, linestyle="--", marker="o", markersize=4,
                      linewidth=1.4, label=f"{LABELS[language]} — validation")

    axis.set_title("Reasoning finetuning: masked cross-entropy on completion "
                   "tokens")
    axis.set_xlabel("optimizer step (64 sequences per step)")
    axis.set_ylabel("cross-entropy (nats / completion token)")
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_val_em(runs: dict[str, tuple[list[dict], list[dict]]],
                baselines: dict[str, float], out_path: Path) -> Path:
    """Validation exact match against step, with the base-model floor marked.

    Args:
        runs: Language to ``(train_records, val_records)``.
        baselines: Language to un-finetuned exact match.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for language, (_, val) in sorted(runs.items()):
        if not val:
            continue
        colour = COLOURS[language]
        axis.plot([r["step"] for r in val],
                  [100.0 * r["val_em"] for r in val], color=colour,
                  marker="o", markersize=4, linewidth=1.8,
                  label=f"{LABELS[language]} — finetuned")
        floor = baselines.get(language)
        if floor is not None:
            axis.axhline(100.0 * floor, color=colour, linestyle=":",
                         linewidth=1.2,
                         label=f"{LABELS[language]} — pretrained base "
                               f"({100.0 * floor:.1f}%)")

    axis.set_title("In-distribution validation exact match (depth 2–3, "
                   "500 held-out examples)")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("exact match (%)")
    axis.set_ylim(-2, 102)
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_schedule(runs: dict[str, tuple[list[dict], list[dict]]],
                  out_path: Path) -> Path:
    """Learning-rate schedule and gradient norm, twin axes.

    Args:
        runs: Language to ``(train_records, val_records)``.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    figure, axis = plt.subplots(figsize=(9, 5.5))
    twin = axis.twinx()
    for language, (train, _) in sorted(runs.items()):
        colour = COLOURS[language]
        steps = [r["step"] for r in train]
        axis.plot(steps, [r["lr"] for r in train], color=colour,
                  linewidth=1.8, label=f"{LABELS[language]} — learning rate")
        norms = [r["grad_norm"] for r in train]
        twin.plot(steps, smooth(norms), color=colour, linestyle="--",
                  alpha=0.7, linewidth=1.2,
                  label=f"{LABELS[language]} — grad norm (smoothed)")

    axis.set_title("Learning-rate schedule and gradient norm")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("learning rate")
    twin.set_ylabel("gradient L2 norm (pre-clip)")
    handles = axis.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    axis.legend(handles[0] + extra[0], handles[1] + extra[1],
                loc="upper right", fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_resume_overlay(gate: dict, out_path: Path) -> Path:
    """Gate P4's overlay: the reference curve and the resumed curve.

    Args:
        gate: The P4 gate result, carrying both loss sequences.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    numbers = gate["numbers"]
    reference = numbers["reference_curve"]
    resumed = numbers["resumed_curve"]
    kill_at = numbers["kill_at"]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot([r["step"] for r in reference], [r["loss"] for r in reference],
              color="#1f77b4", marker="o", markersize=5, linewidth=1.8,
              label="uninterrupted reference run")
    axis.plot([r["step"] for r in resumed], [r["loss"] for r in resumed],
              color="#ff7f0e", marker="x", markersize=9, linewidth=1.4,
              linestyle="--", label="resumed from checkpoint")
    axis.axvline(kill_at, color="grey", linestyle=":", linewidth=1.4,
                 label=f"checkpoint written at step {kill_at}")

    delta = numbers["max_loss_delta"]
    axis.set_title(f"Gate P4 — kill and resume: curves coincide "
                   f"(max |Δloss| = {delta:.1e})")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("cross-entropy (nats / completion token)")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_lr_sweep(sweep: dict, out_path: Path) -> Path:
    """The mini-sweep: validation exact match against peak learning rate.

    Args:
        sweep: The sweep report.
        out_path: Destination PNG.

    Returns:
        The path written.
    """
    runs = sorted(sweep["runs"], key=lambda r: r["peak_lr"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([r["peak_lr"] for r in runs],
              [100.0 * r["best_val_em"] for r in runs],
              color="#2ca02c", marker="o", markersize=7, linewidth=1.8,
              label="best validation exact match")
    winner = sweep["winner"]["peak_lr"]
    axis.axvline(winner, color="grey", linestyle=":", linewidth=1.4,
                 label=f"selected peak LR = {winner:g}")
    axis.set_xscale("log")
    axis.set_title(f"Finetuning learning-rate mini-sweep "
                   f"({sweep['language']}, "
                   f"{sweep['fraction_of_epoch']:g} epoch per point)")
    axis.set_xlabel("peak learning rate (log scale)")
    axis.set_ylabel("best validation exact match (%)")
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def main() -> None:
    """Render every figure that has data available."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", action="append", default=[],
                        metavar="ARCH=LOG",
                        help="finetune log.jsonl per architecture (repeatable); "
                             "defaults to the GPT-ft baseline log")
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--run-prefix", default="sft",
                        help="prefix for the figure file names")
    args = parser.parse_args()

    root = args.root.resolve()
    figures = args.figures or root / "reports" / "gpt_finetune" / "figures"
    report_dir = root / "reports" / "gpt_finetune"

    runs = {}
    items = args.run or [f"gpt={report_dir / 'train_log.jsonl'}"]
    for item in items:
        arch, _, log = item.partition("=")
        log = Path(log)
        if log.exists():
            runs[arch] = read_log(log)
            print(f"read {log}")

    baselines = {}
    baseline_path = report_dir / "sft_baseline.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baselines = {e["language"]: e["val_em"] for e in payload["languages"]}

    written = []
    if runs:
        written.append(plot_loss_curves(
            runs, figures / f"{args.run_prefix}_loss_curves.png"))
        written.append(plot_val_em(
            runs, baselines, figures / f"{args.run_prefix}_val_em.png"))
        written.append(plot_schedule(
            runs, figures / f"{args.run_prefix}_lr_gradnorm.png"))
    else:
        print("no run logs found -- skipping training curves")

    gates_path = report_dir / "sft_gates.json"
    if gates_path.exists():
        payload = json.loads(gates_path.read_text(encoding="utf-8"))
        for entry in payload["languages"]:
            p4 = next((g for g in entry["gates"] if g["gate"] == "P4"), None)
            if p4 and "reference_curve" in p4["numbers"]:
                written.append(plot_resume_overlay(
                    p4, figures / f"sft_gate_p4_resume_{entry['language']}.png"))

    for language in ("hindi",):
        sweep_path = report_dir / f"sft_lr_sweep_{language}.json"
        if sweep_path.exists():
            sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
            written.append(plot_lr_sweep(
                sweep, figures / f"sft_lr_sweep_{language}.png"))

    print()
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
