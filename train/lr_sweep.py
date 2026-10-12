"""Part 3.5 -- the learning-rate sweep, the only hyperparameter search.

WHAT VARIES AND WHAT DOES NOT
-----------------------------
``peak_lr`` varies.  Nothing else does.  Architecture, effective batch, weight
decay, clipping, warmup fraction and the token budget are identical across all
four runs, and identical to what the production run in Part 4 will use.

That is not tidiness, it is the reason the sweep is worth running.  The optimal
learning rate is a property of the *whole* setup: change the effective batch and
the best LR moves roughly with it; change the architecture's depth or width and
the curvature changes with it.  If any of those drifted between the sweep and
the real run, the winner would be the best LR for a configuration that never
trains, and the search would have measured nothing transferable.

WHY AN EQUAL TOKEN BUDGET RATHER THAN EQUAL WALL-CLOCK
------------------------------------------------------
Every run sees the same ~40M tokens (1,200 steps x 32,768).  Comparing at equal
tokens compares the thing that matters -- how much the model learned per unit of
data -- rather than which run happened to get a less contended GPU.

WHY THE COSINE IS COMPRESSED, NOT TRUNCATED
-------------------------------------------
Each run decays over its own 1,200-step horizon rather than running the first
1,200 steps of a 15,000-step schedule.  A truncated schedule would leave every
run stopped near its peak LR, still mid-warmup in effect, and the final losses
would mostly measure noise.  Compressing gives each run a complete
warmup-peak-decay shape, so its final loss is a fair small-scale image of what
that LR does over a full run.

WHY HINDI ONLY
--------------
Both models share an architecture, a vocabulary size and an effective batch, so
the optimal LR is a property of that shared setup far more than of the language.
Sweeping both would double the cost to re-derive the same answer; the plan
instead confirms the winner with the single production run.

THE DECISION RULE (state this in the report)
--------------------------------------------
Pick the LR with the lowest *stable* final validation loss at the equal budget.
Any run that NaNs or spikes is disqualified regardless of where it ended --
instability at 40M tokens becomes divergence at 500M.  If the winner sits at an
edge of the grid, the grid was wrong and gets extended one notch.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from model.config import ModelConfig, Paths, TrainConfig
from train.pretrain import Trainer
from viz.lr_sweep_plots import plot

LOGGER = logging.getLogger("sweep")

#: The GPT baseline's combined grid (rounds 1+2, winner 3e-3 interior).  The
#: LLaMA sweep reuses it exactly; an edge winner is extended one notch.
DEFAULT_GRID = (1.5e-3, 3e-3, 4.5e-3, 6e-3)

#: ~40M tokens at 32,768 tokens/step.
SWEEP_STEPS = 1200

#: 12.5% of the sweep's 1,200-step horizon.  NOTE this is *not* the same
#: fraction the production run uses: 750 steps of a ~15,500-step run is 4.8%.
#: The fraction deliberately differs.  What has to be held constant for the
#: swept LR to transfer is the peak LR and the effective batch, not the warmup
#: length -- warmup guards the unstable first steps, and a short run needs
#: proportionally more of it because its cosine decays so much faster.
SWEEP_WARMUP = 150


def is_stable(history: list[dict]) -> tuple[bool, str]:
    """Decide whether a run trained stably.

    Args:
        history: Per-step records from the run log.

    Returns:
        ``(stable, reason)``; reason is empty when stable.
    """
    losses = [r["train_loss"] for r in history]
    if any(math.isnan(v) or math.isinf(v) for v in losses):
        return False, "loss went NaN or inf"

    # A spike late in training means the LR is too high for the curvature the
    # model has reached, which only gets worse over a longer run.
    tail = losses[len(losses) // 2:]
    if tail and max(tail) > min(tail) + 1.0:
        return False, (f"loss spiked by {max(tail) - min(tail):.2f} nats "
                       "in the second half")
    return True, ""


def run_dir_name(peak_lr: float) -> str:
    """Directory name for one sweep run.

    Uses ``.6g`` rather than ``.0e``.  ``.0e`` rounds to a single significant
    digit, so 1.5e-3 and 2e-3 both render as ``2e-03`` -- two grid points would
    share a directory, and the second would "resume" the first's finished
    checkpoint, skip training entirely and report the first one's loss.  That
    happened; hence this function and the uniqueness assertion in main().

    Args:
        peak_lr: The learning rate for the run.

    Returns:
        A directory name unique to that learning rate.
    """
    return f"lr_{peak_lr:.6g}"


def run_one(peak_lr: float, model_config: ModelConfig, base: TrainConfig,
            paths: Paths, device: str, out_root: Path, steps: int,
            warmup: int, arch: str = "gpt") -> dict:
    """Train one sweep configuration.

    Args:
        peak_lr: The learning rate under test.
        model_config: Architecture, identical for every run.
        base: Recipe to derive from.
        paths: Data locations.
        device: Torch device.
        out_root: Parent directory for per-LR run directories.
        steps: Token budget in steps.
        warmup: Warmup steps.

    Returns:
        Summary of the run, including stability.
    """
    config = TrainConfig.from_dict({
        **base.to_dict(),
        "peak_lr": peak_lr, "max_steps": steps, "warmup_steps": warmup,
        "run_name": f"sweep_{run_dir_name(peak_lr)}",
        # Evaluate often: with only 1,200 steps the shape of the curve is the
        # evidence, not just its endpoint.
        "eval_interval": 100, "ckpt_interval": 600,
    })

    out_dir = out_root / run_dir_name(peak_lr)
    LOGGER.info("\n%s\nsweep run: peak_lr=%.1e -> %s\n%s",
                "=" * 70, peak_lr, out_dir, "=" * 70)

    trainer = Trainer(model_config, config, paths, device, out_dir=out_dir,
                      arch=arch)
    if trainer.maybe_resume():
        # Guard against resuming somebody else's run: a checkpoint carrying a
        # different peak LR means this directory belongs to another grid point.
        resumed_lr = TrainConfig.from_dict(
            torch.load(out_dir / "last.pt", map_location="cpu",
                       weights_only=False)["train_config"]).peak_lr
        if resumed_lr != peak_lr:
            raise RuntimeError(
                f"{out_dir} holds a checkpoint for peak_lr={resumed_lr:.3e}, "
                f"but this run is peak_lr={peak_lr:.3e}")
    summary = trainer.run()

    history = [json.loads(line)
               for line in trainer.log_path.read_text().splitlines() if line]
    stable, reason = is_stable(history)
    summary["stable"] = stable
    summary["instability"] = reason
    summary["val_curve"] = [{"step": r["step"], "val_loss": r["val_loss"]}
                            for r in history if "val_loss" in r]
    return summary


def choose_winner(results: list[dict], grid: tuple[float, ...]) -> dict:
    """Apply the decision rule to the finished runs.

    Args:
        results: Per-run summaries.
        grid: The learning rates that were tried.

    Returns:
        The decision, including whether the grid needs extending.
    """
    stable = [r for r in results if r["stable"]]
    if not stable:
        return {"winner": None,
                "reason": "every run was unstable; the whole grid is too high"}

    best = min(stable, key=lambda r: r["final_val_loss"])
    lo, hi = min(grid), max(grid)
    at_edge = best["peak_lr"] in (lo, hi)

    decision = {
        "winner": best["peak_lr"],
        "final_val_loss": best["final_val_loss"],
        "discarded_unstable": [r["peak_lr"] for r in results if not r["stable"]],
        "at_grid_edge": at_edge,
    }
    if at_edge:
        # The optimum may lie beyond the grid, so the measured best is only a
        # boundary, not a maximum.  Derived from the grid rather than hardcoded:
        # a fixed 1.5e-3 was written for the original {2e-4..1e-3} grid and
        # became nonsense once the grid was extended past it (it proposed
        # "extending" a 3e-3 winner *downward*).
        extension = round(lo / 2, 12) if best["peak_lr"] == lo else round(hi * 1.5, 12)
        decision["extend_grid_with"] = extension
        decision["reason"] = (
            f"winner {best['peak_lr']:.1e} sits at the "
            f"{'low' if best['peak_lr'] == lo else 'high'} edge of the grid; "
            f"the plan says extend one notch to {extension:.1e} and re-check")
    else:
        decision["reason"] = (
            f"{best['peak_lr']:.1e} has the lowest stable validation loss at "
            "the equal 40M-token budget, with a worse neighbour on each side")
    return decision


def main() -> None:
    """Run the sweep and record the winner."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--arch", default="gpt",
                        help="Decoder architecture: gpt (llama from Stage 2)")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="Architecture config; defaults to the variant's")
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--steps", type=int, default=SWEEP_STEPS)
    parser.add_argument("--warmup", type=int, default=SWEEP_WARMUP)
    parser.add_argument("--device", default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=Path("reports/gpt_pretrain/figures"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/lr_sweep.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    paths = Paths.for_variant(args.language, args.root, args.variant)
    model_config = ModelConfig.load(args.model_config or paths.model_config)
    base = TrainConfig.load(paths.train_config)
    overrides = {k: v for k, v in (("micro_batch", args.micro_batch),
                                   ("grad_accum", args.grad_accum))
                 if v is not None}
    if overrides:
        base = TrainConfig.from_dict({**base.to_dict(), **overrides})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = args.out or (paths.artifacts / "runs" / f"lr_sweep_{args.arch}")

    names = [run_dir_name(lr) for lr in args.grid]
    if len(set(names)) != len(names):
        raise SystemExit(
            f"grid {args.grid} produces duplicate run directories {names}; "
            "two runs would share a checkpoint")

    tokens = base.tokens_per_step(model_config.context_len) * args.steps
    LOGGER.info("sweep on %s: %d runs x %s tokens each, device=%s",
                args.language, len(args.grid), f"{tokens:,}", device)

    results = [run_one(lr, model_config, base, paths, device, out_root,
                       args.steps, args.warmup, args.arch)
               for lr in args.grid]
    decision = choose_winner(results, tuple(args.grid))

    LOGGER.info("\n%s\nSWEEP RESULTS\n%s", "=" * 70, "=" * 70)
    LOGGER.info("%-10s %14s %12s %s", "peak_lr", "final val", "stable", "note")
    for result in sorted(results, key=lambda r: r["peak_lr"]):
        LOGGER.info("%-10.1e %14.4f %12s %s", result["peak_lr"],
                    result["final_val_loss"], result["stable"],
                    result["instability"])
    LOGGER.info("\nWINNER: %s", decision.get("winner"))
    LOGGER.info("REASON: %s", decision["reason"])

    subtitle = (f"Hindi -- {args.arch}, "
                f"{args.steps:,} steps x "
                f"{base.tokens_per_step(model_config.context_len):,} tokens = "
                f"{tokens / 1e6:.0f}M tokens per run")
    figures = plot(results, decision, args.figures, subtitle, paths.suffix)
    report = {"language": args.language, "variant": args.variant,
              "arch": args.arch,
              "model_config": model_config.to_dict(),
              "grid": args.grid,
              "steps_per_run": args.steps, "warmup": args.warmup,
              "tokens_per_run": tokens,
              "effective_batch_tokens":
                  base.tokens_per_step(model_config.context_len),
              "results": results, "decision": decision,
              "figures": [str(p) for p in figures]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("report -> %s", args.report)
    LOGGER.info("figures -> %s", ", ".join(str(p) for p in figures))


if __name__ == "__main__":
    main()
