"""Verify the fixed GPT baselines reproduce their recorded numbers from this layout.

    python -m tests.verify_gpt_baselines            # both checks
    python -m tests.verify_gpt_baselines --skip-lm  # held-out EM only

Two checks, both against values recorded when the checkpoints were produced:

1. **Pretrained GPT** (``artifacts/checkpoints/gpt/final_16k.pt``): mean test
   cross-entropy over the full 16K test split, each token scored once, must
   reproduce 3.4080 nats (PPL 30.20, BPB 0.4892).
2. **GPT-ft** (``artifacts/checkpoints/gpt_ft/ft_60k/best.pt``): exact match on
   the fixed 500-example subset of the held-out-name validation file must
   reproduce 0.954, using the finetuning trainer's own ``validate_em``.  The
   training split is tokenized on the way, so its token counts are checked
   against the run's ``summary.json`` too -- that confirms
   ``train_60k.jsonl`` is the file the course run trained on.

The GPT models are never retrained; this only evaluates them.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from model.config import Paths
from train.finetune import SFTTrainer
from train.finetune_config import SFTConfig

LOGGER = logging.getLogger("tests.verify_gpt_baselines")

#: Recorded in reports/gpt_pretrain/eval_16k.json (full test split).
RECORDED_TEST = {"mean_ce_nats": 3.4079521466706377,
                 "perplexity": 30.20332889660831,
                 "bits_per_byte": 0.489164152925853}
#: Recorded in the GPT-ft checkpoint and its summary.json.
RECORDED_EM = 0.954
#: Recorded train_stats of the course finetune (summary.json).
RECORDED_TRAIN_STATS = {"examples": 60000, "tokens_total": 5782373,
                        "tokens_supervised": 3523166}


def check_test_loss(root: Path, device: str, batch_size: int) -> dict:
    """Score the pretrained GPT on the full test split and compare.

    Args:
        root: Repository root.
        device: Torch device.
        batch_size: Sequences per forward pass.

    Returns:
        Measured values, recorded values and absolute differences.
    """
    from eval.lm_metrics import evaluate

    result = evaluate("hindi", "test", device=device, batch_size=batch_size,
                      root=root, variant="16k")
    diffs = {k: abs(result[k] - v) for k, v in RECORDED_TEST.items()}
    return {"measured": {k: result[k] for k in RECORDED_TEST},
            "recorded": RECORDED_TEST, "abs_diff": diffs,
            "tokens_scored": result["tokens_scored"],
            "passed": diffs["mean_ce_nats"] < 5e-4}


def check_heldout_em(root: Path, device: str) -> dict:
    """Score GPT-ft on the fixed held-out-name EM subset and compare.

    Args:
        root: Repository root.
        device: Torch device.

    Returns:
        Measured EM, recorded EM, and the training-split token-count check.
    """
    paths = Paths.for_variant("hindi", root, "16k")
    config = SFTConfig.load(root / "configs" / "finetune_shared.yaml")
    out_dir = paths.artifacts / "verify" / "gpt_ft_em"
    trainer = SFTTrainer(config, paths, device, out_dir, paths.finetune_data,
                         arch="gpt", base=paths.gpt_finetuned)
    em = trainer.validate_em()
    train_stats = {k: trainer.train_stats[k] for k in RECORDED_TRAIN_STATS}
    return {"measured_em": em["exact_match"], "correct": em["correct"],
            "total": em["total"], "recorded_em": RECORDED_EM,
            "abs_diff": abs(em["exact_match"] - RECORDED_EM),
            "train_stats": train_stats,
            "train_stats_match": train_stats == RECORDED_TRAIN_STATS,
            "passed": (abs(em["exact_match"] - RECORDED_EM) < 1e-9
                       and train_stats == RECORDED_TRAIN_STATS)}


def main() -> None:
    """Run both checks and write a JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--skip-em", action="store_true")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/verify_gpt_baselines.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"device": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "torch": torch.__version__}
    if not args.skip_lm:
        report["pretrained_test_loss"] = check_test_loss(root, device,
                                                         args.batch_size)
        LOGGER.info("test loss: %s", report["pretrained_test_loss"])
    if not args.skip_em:
        report["finetuned_heldout_em"] = check_heldout_em(root, device)
        LOGGER.info("held-out EM: %s", report["finetuned_heldout_em"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(v.get("passed", True) for v in report.values()
               if isinstance(v, dict)):
        raise SystemExit("VERIFICATION FAILED")


if __name__ == "__main__":
    main()
