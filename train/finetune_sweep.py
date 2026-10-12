"""Learning-rate mini-sweep for the finetune (§3), mirroring Phase 2's discipline.

    python -m train.finetune_sweep --language hindi --grid 1e-4 2e-4 3e-4

WHY THE SAME SWEEP PROTOCOL FOR EVERY ARCHITECTURE
--------------------------------------------------
An architecture comparison is only meaningful at a matched tuning budget: if
one model got a wider or finer search, a difference in reasoning accuracy could
be a difference in how well each run was tuned rather than a fact about the
architectures.  So every architecture is swept with the same grid, the same
fraction of an epoch and the same selection rule.

WHY val-EM DECIDES AND NOT val LOSS
-----------------------------------
Token loss rewards fluent continuation of the chain of thought; exact match
rewards getting the final label right.  They can disagree -- a run can produce
a confident, well-formed chain that reaches the wrong conclusion.  Exact match
is what the task is scored on, so it is what selects.  Both are reported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from model.config import Paths
from train.finetune_config import SFTConfig
from train.finetune import SFTTrainer

LOGGER = logging.getLogger("train.finetune_sweep")


def run_one(language: str, peak_lr: float, fraction: float, root: Path,
            device: str, out_root: Path, base_config: SFTConfig,
            arch: str = "gpt", base_ckpt: Path | None = None) -> dict:
    """Run one short finetune at a given peak learning rate.

    Args:
        language: Language to sweep on.
        peak_lr: Peak learning rate under test.
        fraction: Fraction of one epoch to run.
        root: Repository root.
        device: Torch device.
        out_root: Parent directory for this run's output.
        base_config: Recipe to copy the non-LR hyperparameters from.
        arch: Decoder architecture.
        base_ckpt: Pretrained checkpoint to start from.

    Returns:
        The run's outcome, including its best val-EM.
    """
    import dataclasses

    config = dataclasses.replace(
        base_config, language=language, peak_lr=peak_lr,
        run_name=f"sweep_{peak_lr:g}")
    paths = Paths.for_variant(language, root, config.variant)
    out_dir = out_root / f"lr_{peak_lr:g}"

    trainer = SFTTrainer(config, paths, device, out_dir, paths.finetune_data,
                         arch=arch, base=base_ckpt)
    # The schedule must see the FULL planned horizon even though only a
    # fraction is run: a cosine compressed into half an epoch would anneal to
    # its floor and flatter the short run, which is not what the full run will
    # experience.
    steps = max(1, int(fraction * trainer.steps_per_epoch))
    config.save(out_dir / "config.json")

    started = time.time()
    summary = trainer.run(max_steps=steps)
    final_val = trainer.validate()
    final_em = trainer.validate_em()

    return {
        "peak_lr": peak_lr,
        "steps": trainer.step,
        "planned_total_steps": trainer.total_steps,
        "fraction_of_epoch": fraction,
        "best_val_em": trainer.best_em,
        "best_step": trainer.best_step,
        "final_val_loss": final_val["val_loss"],
        "final_val_em": final_em["exact_match"],
        "final_no_answer_cue": final_em["no_answer_cue"],
        "final_hit_length_cap": final_em["hit_length_cap"],
        "wall_clock_seconds": round(time.time() - started, 1),
        "out_dir": str(out_dir),
        "diverged": trainer.best_em < 0.0,
    }


def pick_winner(results: list[dict], em_subset: int = 500) -> dict:
    """Choose a learning rate, preferring a bracketed point among the tied.

    WHY THE ARGMAX IS THE WRONG ANSWER HERE
    ---------------------------------------
    Exact match is a proportion measured on a finite subset, so it carries
    sampling error: at ``n = 500`` and ``p ~ 0.8`` one standard error is about
    1.8 percentage points.  Several learning rates typically land inside that
    band, and picking the raw maximum then just picks the luckiest sample --
    and, worse, often picks the highest rate on the grid, whose optimum is
    unbracketed and may lie outside it entirely.  Phase 2's report criticises
    its own 24K baseline for exactly that.

    So the rule is: take every rate within one standard error of the best,
    and among those prefer one that is BRACKETED -- both grid neighbours
    measurably worse -- which is evidence of a real interior optimum rather
    than a grid edge.  Ties break toward the lower rate, the more conservative
    choice for a full-length run.

    Args:
        results: One entry per swept learning rate.
        em_subset: Examples the exact match was measured on, for the error bar.

    Returns:
        The decision, with the reasoning recorded alongside it.
    """
    import math

    ordered = sorted(results, key=lambda r: r["peak_lr"])
    best_em = max(r["best_val_em"] for r in ordered)
    standard_error = math.sqrt(max(best_em * (1 - best_em), 1e-9) / em_subset)
    tied = [r for r in ordered if best_em - r["best_val_em"] <= standard_error]

    def bracketed(entry: dict) -> bool:
        """True if both grid neighbours scored strictly lower."""
        index = ordered.index(entry)
        if index == 0 or index == len(ordered) - 1:
            return False
        return (ordered[index - 1]["best_val_em"] < entry["best_val_em"]
                and ordered[index + 1]["best_val_em"] < entry["best_val_em"])

    candidates = [r for r in tied if bracketed(r)]
    if candidates:
        chosen = min(candidates, key=lambda r: r["peak_lr"])
        reason = (f"within one standard error ({standard_error:.4f}) of the "
                  f"best on the grid, and bracketed -- both neighbours score "
                  f"lower -- so this is an interior optimum rather than a "
                  f"grid edge")
    else:
        chosen = min(tied, key=lambda r: r["peak_lr"])
        reason = (f"lowest rate within one standard error "
                  f"({standard_error:.4f}) of the best; NO candidate in that "
                  f"band is bracketed, so the optimum is not enclosed by this "
                  f"grid")

    return {
        "peak_lr": chosen["peak_lr"],
        "best_val_em": chosen["best_val_em"],
        "argmax_peak_lr": max(ordered, key=lambda r: r["best_val_em"])["peak_lr"],
        "argmax_val_em": best_em,
        "standard_error": round(standard_error, 5),
        "em_subset": em_subset,
        "tied_within_one_se": [r["peak_lr"] for r in tied],
        "bracketed": bracketed(chosen),
        "reason": reason,
        "ranking": [{"peak_lr": r["peak_lr"], "best_val_em": r["best_val_em"],
                     "final_val_loss": r["final_val_loss"],
                     "bracketed": bracketed(r)}
                    for r in sorted(ordered,
                                    key=lambda r: -r["best_val_em"])],
    }


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--arch", default="gpt")
    parser.add_argument("--base", type=Path, default=None,
                        help="Pretrained checkpoint to finetune from")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_shared.yaml"))
    parser.add_argument("--grid", nargs="+", type=float,
                        default=[1e-4, 2e-4, 3e-4])
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="fraction of one epoch per sweep run")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--em-file", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.root.resolve()
    out_root = args.out or root / "artifacts" / "runs" / f"finetune_sweep_{args.arch}"
    report = (args.report
              or root / "reports" / "gpt_finetune" / f"sft_lr_sweep_{args.language}.json")

    base = SFTConfig.load(args.config)
    if args.micro_batch is not None:
        base.micro_batch = args.micro_batch
    if args.grad_accum is not None:
        base.grad_accum = args.grad_accum
    for attr, value in (("em_file", args.em_file),
                        ("epochs", args.epochs)):
        if value is not None:
            setattr(base, attr, value)

    results = []
    for peak_lr in args.grid:
        LOGGER.info("sweeping peak_lr=%g", peak_lr)
        results.append(run_one(args.language, peak_lr, args.fraction, root,
                               device, out_root, base, args.arch, args.base))
        LOGGER.info("  best val-EM %.4f", results[-1]["best_val_em"])

    winner = pick_winner(results, base.em_subset)
    payload = {"language": args.language, "device": device,
               "grid": args.grid, "fraction_of_epoch": args.fraction,
               "effective_batch": base.effective_batch,
               "runs": results, "winner": winner}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"| peak LR | best val-EM | final val loss | steps |")
    print(f"| --- | ---: | ---: | ---: |")
    for entry in results:
        print(f"| {entry['peak_lr']:g} | {entry['best_val_em']:.4f} | "
              f"{entry['final_val_loss']:.4f} | {entry['steps']} |")
    print()
    print(f"winner: peak_lr={winner['peak_lr']:g} "
          f"(val-EM {winner['best_val_em']:.4f}) -- {winner['reason']}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
