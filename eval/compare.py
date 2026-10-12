"""Compare training recipes at an equal token budget, in bits per byte.

WHY THIS MODULE EXISTS
----------------------
The baseline ran for 15,521 steps (491.5M tokens).  Model3's comparison run is
4,500 steps (147.5M tokens), because the quota will not stretch to a second full
run per candidate.  Putting the two finished numbers side by side would compare
a budget difference and call it an architecture difference -- which is the one
mistake this whole comparison exists to avoid.

There is an honest reading available at no extra GPU cost.  Both runs log
validation loss every 250 steps, so the baseline's loss *at step 4,500* is
already on disk.  Comparing the two at the same step compares two architectures
that have seen the same number of tokens through the same effective batch.

WHY THE COMPARISON MUST BE IN BITS PER BYTE
-------------------------------------------
The baseline's tokens come from a 24,000-token vocabulary and Model3's from a
16,000-token one.  A 16K token covers less text -- 4.09 characters against 4.25
for Hindi -- so the two models are not answering the same exam, and their
per-token losses are not comparable.  Nats per token flatters whichever model
has the smaller chunks to predict.

Bits per byte re-denominates the same total uncertainty in a unit both share::

    BPB = total_CE_nats / (ln 2 x UTF-8 bytes of the same text)

and since ``total_CE = mean_CE x n_tokens``, a mean CE converts directly::

    BPB = mean_CE / (ln 2 x bytes_per_token)

``bytes_per_token`` is a property of the tokenizer and the split, measured here
by decoding the split back to text and counting its UTF-8 bytes -- the same
denominator ``eval.lm_metrics`` uses for the definitive Part 5 numbers.

WHAT THIS IS NOT
----------------
These are *validation* numbers taken from a 50-batch running estimate, not the
full sequential pass over the test split that Part 5 makes.  They are the right
tool for reading two curves against each other; the reported headline BPB still
comes from ``eval.lm_metrics``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from model.config import ModelConfig, Paths

LOGGER = logging.getLogger("compare")

#: Tokens decoded per call, matching ``eval.lm_metrics``.  SentencePiece drops a
#: leading word-boundary marker at each chunk edge, which across tens of
#: millions of tokens is a rounding error, and keeps peak memory bounded.
DECODE_CHUNK = 1_000_000


def bytes_per_token(language: str, split: str, variant: str,
                    root: Path = Path("."), sample_tokens: int | None = None
                    ) -> dict:
    """Measure how many UTF-8 bytes one token of this split is worth.

    Args:
        language: ``"hindi"`` (the only language).
        split: Which split to measure; use the one the losses were computed on.
        variant: Architecture variant, which selects the tokenizer and bin.
        root: Repository root.
        sample_tokens: Measure only the first N tokens.  The ratio is a
            corpus-level constant, so a large prefix gives the same number as
            the whole file at a fraction of the decode time; ``None`` reads all
            of it.

    Returns:
        The measured ratio with the counts it came from.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, variant)
    tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
    tokens = np.memmap(paths.split_bin(split), dtype=np.uint16, mode="r")
    if sample_tokens is not None:
        tokens = tokens[:sample_tokens]

    total_bytes = 0
    for start in range(0, len(tokens), DECODE_CHUNK):
        chunk = np.asarray(tokens[start:start + DECODE_CHUNK], dtype=np.int64)
        total_bytes += len(tokenizer.decode(chunk.tolist()).encode("utf-8"))

    return {
        "language": language,
        "variant": variant,
        "split": split,
        "vocab_size": paths.vocab_size,
        "tokenizer": str(paths.tokenizer),
        "tokens_measured": int(len(tokens)),
        "utf8_bytes": total_bytes,
        "bytes_per_token": total_bytes / len(tokens),
    }


def bpb_from_ce(mean_ce_nats: float, bytes_per_tok: float) -> float:
    """Convert a mean per-token cross-entropy into bits per byte.

    Args:
        mean_ce_nats: Mean cross-entropy in nats per token.
        bytes_per_tok: UTF-8 bytes each token is worth, from
            :func:`bytes_per_token`.

    Returns:
        Bits per byte.
    """
    return mean_ce_nats / (math.log(2) * bytes_per_tok)


def val_curve(log_path: Path) -> list[dict]:
    """Read the validation points out of a run log.

    Args:
        log_path: Path to a run's ``log.jsonl``.

    Returns:
        ``{"step", "val_loss", "lr"}`` records, in order.  The learning rate is
        carried because it is what reveals *where on its schedule* a run was at
        a given step -- see :func:`anneal_state`.
    """
    points = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "val_loss" in record:
            points.append({"step": record["step"],
                           "val_loss": record["val_loss"],
                           "lr": record.get("lr")})
    return points


def at_step(points: list[dict], step: int) -> dict | None:
    """The validation point nearest ``step``, or None if the run never got there.

    Args:
        points: Output of :func:`val_curve`.
        step: The step to read at.

    Returns:
        The nearest recorded point, or ``None`` if the run stopped short of it.
    """
    reached = [p for p in points if p["step"] <= step]
    if not reached:
        return None
    return min(reached, key=lambda p: abs(p["step"] - step))


def val_curve_from_sweep(report_path: Path, peak_lr: float) -> list[dict]:
    """Recover one sweep run's validation curve from a sweep report.

    WHY A SWEEP RUN IS THE CLEANEST COMPARISON AVAILABLE
    ----------------------------------------------------
    Sweep runs compress a full warmup-peak-decay cosine into their own short
    horizon, so each one *finishes annealed*.  Two sweep runs at the same steps,
    warmup, batch and peak LR therefore differ only in architecture -- they are
    matched on the schedule position that :func:`anneal_state` exists to warn
    about, which the production logs are not.

    The learning rate is not stored per-point in a sweep report, so it is
    recomputed with :meth:`TrainConfig.lr_at`.  That is exact rather than an
    approximation: the trainer sets the LR from that same pure function of the
    step number, which is the property that makes resume exact.

    Args:
        report_path: A ``phase2_lr_sweep*.json`` written by :mod:`train.lr_sweep`.
        peak_lr: Which run in the grid to extract.

    Returns:
        ``{"step", "val_loss", "lr"}`` records, in order.

    Raises:
        SystemExit: If the report holds no run at that learning rate.
    """
    from model.config import TrainConfig

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    matches = [r for r in report["results"]
               if math.isclose(r["peak_lr"], peak_lr, rel_tol=1e-9)]
    if not matches:
        tried = ", ".join(f"{r['peak_lr']:.3g}" for r in report["results"])
        raise SystemExit(
            f"{report_path} has no run at peak_lr={peak_lr:.3g}; it holds {tried}")

    schedule = TrainConfig(peak_lr=peak_lr,
                           warmup_steps=report["warmup"],
                           max_steps=report["steps_per_run"])
    return [{"step": p["step"], "val_loss": p["val_loss"],
             "lr": schedule.lr_at(p["step"])}
            for p in matches[0]["val_curve"]]


def anneal_state(points: list[dict], point: dict) -> dict:
    """How far through its learning-rate schedule a run was at one point.

    THE BIAS THIS EXISTS TO EXPOSE
    ------------------------------
    Reading two runs at a matched step equalises the *tokens* they have seen.
    It does not equalise where each is on its cosine.  A short run whose whole
    schedule fits inside that step count has fully annealed -- its LR has
    decayed to the floor and the weights have settled.  A long run passing
    through the same step is still near its peak LR and still bouncing.

    The annealed model will show the lower loss, and most of that gap is the
    anneal, not the architecture.  Concretely: at step 4,500 the baseline sits
    at 86% of its peak LR while Model3, whose run ends there, is at its 10%
    floor -- so the comparison flatters Model3.

    Reporting the comparison without this number would be presenting a
    schedule artifact as an architecture result.

    Args:
        points: The run's full validation curve.
        point: The point being compared.

    Returns:
        The learning rate there, its share of the run's peak, and whether the
        run had finished by then.
    """
    lrs = [p["lr"] for p in points if p.get("lr") is not None]
    peak = max(lrs) if lrs else None
    lr = point.get("lr")
    final_step = points[-1]["step"] if points else None

    share = (lr / peak) if (lr is not None and peak) else None
    return {
        "lr_at_step": lr,
        "peak_lr_seen": peak,
        "fraction_of_peak_lr": share,
        "run_final_step": final_step,
        "is_runs_final_point": point["step"] == final_step,
        # At or near the floor means the cosine has done its work.
        "fully_annealed": bool(share is not None and share <= 0.15),
    }


def describe(language: str, variant: str, log_path: Path, split: str,
             root: Path = Path("."), sample_tokens: int | None = None,
             points: list[dict] | None = None) -> dict:
    """Summarise one run: its architecture, its curve, and its BPB conversion.

    Args:
        language: ``"hindi"`` (the only language).
        variant: Architecture variant.
        log_path: The run's ``log.jsonl``.
        split: Split the validation loss was measured on.
        root: Repository root.
        sample_tokens: Passed to :func:`bytes_per_token`.
        points: A pre-built validation curve, e.g. from
            :func:`val_curve_from_sweep`.  When given, ``log_path`` is recorded
            for provenance but not read.

    Returns:
        Everything needed to place this run in the comparison table.
    """
    paths = Paths.for_variant(language, root, variant)
    config = ModelConfig.load(paths.model_config)
    ratio = bytes_per_token(language, split, variant, root, sample_tokens)
    points = points if points is not None else val_curve(log_path)

    return {
        "language": language,
        "variant": variant or "baseline",
        "architecture": (f"{config.vocab_size // 1000}K / {config.d_model} / "
                         f"{config.n_layers}L"),
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_k": config.d_k,
        "parameter_count": config.parameter_count,
        "bytes_per_token": ratio["bytes_per_token"],
        "bytes_per_token_measured_on": ratio,
        "log": str(log_path),
        "final": points[-1] if points else None,
        "val_curve": points,
    }


def compare(runs: list[dict], step: int) -> dict:
    """Place every run's loss at a common step and convert it to bits per byte.

    Args:
        runs: Output of :func:`describe`, one per recipe.
        step: The common step to read every curve at.

    Returns:
        The comparison, including which runs could not reach ``step``.
    """
    rows, short = [], []
    for run in runs:
        point = at_step(run["val_curve"], step)
        if point is None:
            short.append(run["variant"])
            continue
        state = anneal_state(run["val_curve"], point)
        rows.append({
            "variant": run["variant"],
            "architecture": run["architecture"],
            "parameter_count": run["parameter_count"],
            "step": point["step"],
            "tokens": point["step"] * 32_768,
            "val_ce_nats_per_token": point["val_loss"],
            "val_ppl": math.exp(min(point["val_loss"], 20)),
            "bytes_per_token": run["bytes_per_token"],
            "val_bpb": bpb_from_ce(point["val_loss"], run["bytes_per_token"]),
            "schedule_position": state,
        })

    rows.sort(key=lambda r: r["val_bpb"])
    return {
        "requested_step": step,
        "rows": rows,
        "runs_that_never_reached_that_step": short,
        "winner": rows[0]["variant"] if rows else None,
        "note": (
            "Validation loss at a matched step, converted to bits per byte so "
            "recipes with different vocabularies are on one axis. Perplexity "
            "is shown per row but must NOT be compared across rows: the 16K "
            "and 24K tokenizers cut the same text into differently sized "
            "pieces, so their per-token losses answer different questions. "
            "These are 50-batch validation estimates; the headline numbers "
            "come from eval.lm_metrics's full sequential pass over the test split."),
        "schedule_warning": _schedule_warning(rows),
    }


def _schedule_warning(rows: list[dict]) -> str | None:
    """State which rows the matched-step comparison favours, and why.

    Args:
        rows: The comparison rows, each carrying a ``schedule_position``.

    Returns:
        A warning naming the advantaged runs, or ``None`` if every run was at
        a comparable point on its schedule.
    """
    annealed = [r["variant"] for r in rows
                if r["schedule_position"]["fully_annealed"]]
    mid = [r for r in rows if not r["schedule_position"]["fully_annealed"]]
    if not annealed or not mid:
        return None

    detail = ", ".join(
        f"{r['variant']} at "
        f"{100 * r['schedule_position']['fraction_of_peak_lr']:.0f}% of peak LR"
        for r in mid if r["schedule_position"]["fraction_of_peak_lr"])
    return (
        f"NOT A LIKE-FOR-LIKE READ. {', '.join(annealed)} had fully annealed by "
        f"this step, while {detail}. An annealed model settles into a lower "
        f"loss than a mid-schedule one at the same token count, so this "
        f"comparison is biased IN FAVOUR OF {', '.join(annealed)}. Read it "
        f"accordingly: if an annealed run still loses, that is strong evidence; "
        f"if it wins narrowly, the anneal alone may account for the gap.")


def main() -> None:
    """Build the equal-budget comparison table and write it to a report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument(
        "--run", action="append", default=[], metavar="VARIANT=LOG",
        help='A run to include, e.g. "=logs/phase2/runs/24k_hindi/log.jsonl" for '
             'the baseline or "16k=logs/phase2/runs/16k_hindi/log.jsonl". '
             "Repeat for each recipe.")
    parser.add_argument(
        "--sweep-run", action="append", default=[], metavar="VARIANT=REPORT@LR",
        help='A sweep run to include, e.g. "16k=report/'
             'phase2_lr_sweep_16k.json@3e-3". Sweep runs finish annealed, so '
             "two of them at matched steps/warmup/batch/LR are the cleanest "
             "architecture comparison available.")
    parser.add_argument("--step", type=int, default=4500,
                        help="The common step to read every curve at; defaults "
                             "to the reduced comparison budget")
    parser.add_argument("--split", default="val",
                        choices=("train", "val", "test"),
                        help="Split the logged loss was measured on")
    parser.add_argument("--sample-tokens", type=int, default=8_000_000,
                        help="Tokens to decode when measuring bytes/token; the "
                             "ratio is a corpus constant, so a large prefix is "
                             "enough. Pass 0 to use the whole split.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/compare.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.run and not args.sweep_run:
        raise SystemExit("pass at least one --run VARIANT=LOG or --sweep-run")
    sample = args.sample_tokens or None

    runs = []
    for spec in args.run:
        variant, _, log = spec.partition("=")
        if not log:
            raise SystemExit(f"--run {spec!r} is not VARIANT=LOG")
        LOGGER.info("reading %s (%s) ...", log, variant or "baseline")
        runs.append(describe(args.language, variant, Path(log), args.split,
                             args.root, sample))

    for spec in args.sweep_run:
        variant, _, rest = spec.partition("=")
        report_path, _, lr_text = rest.partition("@")
        if not report_path or not lr_text:
            raise SystemExit(f"--sweep-run {spec!r} is not VARIANT=REPORT@LR")
        LOGGER.info("reading %s @ %s (%s) ...", report_path, lr_text,
                    variant or "baseline")
        curve = val_curve_from_sweep(Path(report_path), float(lr_text))
        runs.append(describe(args.language, variant, Path(report_path),
                             args.split, args.root, sample, points=curve))

    result = compare(runs, args.step)

    LOGGER.info("\n%s", "=" * 78)
    LOGGER.info("%s -- validation at step %d (%s tokens each)",
                args.language, args.step, f"{args.step * 32768:,}")
    LOGGER.info("%s", "=" * 78)
    LOGGER.info("%-10s %-16s %11s %9s %9s %8s %9s",
                "variant", "architecture", "params", "CE", "PPL",
                "bytes/tok", "BPB")
    LOGGER.info("%s", "-" * 78)
    for row in result["rows"]:
        LOGGER.info("%-10s %-16s %11s %9.4f %9.2f %8.3f %9.4f",
                    row["variant"], row["architecture"],
                    f"{row['parameter_count']:,}",
                    row["val_ce_nats_per_token"], row["val_ppl"],
                    row["bytes_per_token"], row["val_bpb"])
    if result["runs_that_never_reached_that_step"]:
        LOGGER.info("\nnot at step %d yet: %s", args.step,
                    ", ".join(result["runs_that_never_reached_that_step"]))
    LOGGER.info("\nlowest BPB: %s", result["winner"])
    if result.get("schedule_warning"):
        LOGGER.info("\n%s", result["schedule_warning"])

    payload = {"language": args.language, "split": args.split,
               "comparison": result,
               "runs": [{k: v for k, v in r.items() if k != "val_curve"}
                        for r in runs]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
