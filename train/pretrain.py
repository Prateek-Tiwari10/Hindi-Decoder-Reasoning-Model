"""The pretraining loop: AdamW, warmup-then-cosine, fp16, resumable checkpoints.

    python -m train.pretrain --arch gpt          # (the GPT baseline is never retrained)

One script for every architecture.  ``--arch`` picks the decoder; the recipe
(data, order, seed, schedule, batch, precision) comes from
``configs/pretrain_shared.yaml`` and is identical for all of them.

THE RECIPE IS FROZEN
--------------------
Everything here except ``peak_lr`` is fixed by the plan.  The LR sweep varies
that one number and nothing else, and the production run reuses the winner
unchanged.  That discipline is what makes the sweep's result transfer: if the
effective batch or the architecture moved between the sweep and the real run,
the best LR would move with them and the search would have measured nothing.

THE EFFECTIVE BATCH IS THE INVARIANT
------------------------------------
32,768 tokens per optimiser step = 64 sequences x 512 context, realised as
``micro_batch=32`` twice over via gradient accumulation.  If a GPU cannot hold
32, halve it and double ``grad_accum``: the *product* is the optimisation
problem, and the split is only a memory detail.  Gradients accumulate in the
``.grad`` buffers across micro-steps and the optimiser steps once, so the update
is mathematically the same as one 64-sequence batch.

WHY THE LOSS IS DIVIDED BY grad_accum
-------------------------------------
Each micro-batch's loss is already a mean over its own tokens.  Summing two of
them would give twice the gradient of the true mean, which silently doubles the
effective learning rate.  Dividing by ``grad_accum`` before ``backward()``
restores the mean over the whole effective batch.

WHY fp16 AND NOT bf16
---------------------
Kaggle's T4 and P100 are pre-Ampere and have no bf16 path.  fp16 has a much
narrower exponent range, so small gradients underflow to zero -- which is what
``GradScaler`` exists to prevent: it multiplies the loss by a large factor
before ``backward()``, so gradients land inside fp16's representable range, then
unscales them before the optimiser sees them.  The scale is adapted
automatically, backing off whenever an inf or NaN appears.

CLIPPING MUST HAPPEN AFTER UNSCALING
------------------------------------
``scaler.unscale_(optimizer)`` is called before ``clip_grad_norm_``.  Clipping
scaled gradients would compare an arbitrarily inflated norm against the 1.0
threshold and clip essentially every step, destroying the update.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from model.config import ModelConfig, Paths, TrainConfig
from data.token_stream import TokenStream, to_torch
from model import build_model

LOGGER = logging.getLogger("train")


def parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and undecayed groups.

    Weight decay is applied to the 2-D matrices -- embeddings and the linear
    projections -- and withheld from biases and LayerNorm weights.  Those are
    1-D per-feature scales and offsets: shrinking them toward zero does not
    regularise anything, it just fights the normalisation the layer exists to
    provide.

    The tied embedding appears once because ``named_parameters`` de-duplicates
    shared tensors, so it is decayed once rather than twice.

    Args:
        model: The model whose parameters to group.
        weight_decay: Decay for the 2-D group.

    Returns:
        Two parameter-group dicts ready for AdamW.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    """Owns one training run: model, optimiser, schedule and checkpoints."""

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig,
                 paths: Paths, device: str, out_dir: Path | None = None,
                 arch: str = "gpt") -> None:
        """Set up the run.

        Args:
            model_config: Architecture.
            train_config: Recipe.
            paths: Where this language's data lives.
            device: Torch device.
            out_dir: Where checkpoints and logs go.  Defaults to the language's
                ``train/`` directory; the sweep overrides it per LR.
        """
        # A 16K config pointed at the 24K bins would not crash: every 16K id is
        # also a valid 24K id, so the run would train happily on token streams
        # its tokenizer never wrote and only the loss curve would look odd.
        # Catch it here, before a GPU-hour is spent.
        if model_config.vocab_size != paths.vocab_size:
            raise ValueError(
                f"config vocab_size={model_config.vocab_size} but "
                f"{paths.split_bin('train')} was tokenized with "
                f"vocab_size={paths.vocab_size}; pass the matching --variant")

        self.model_config = model_config
        self.train_config = train_config
        self.paths = paths
        self.device = device
        self.out_dir = Path(out_dir or paths.ckpt_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(train_config.seed)
        np.random.seed(train_config.seed)

        self.arch = arch
        self.model = build_model(model_config, device, arch=arch)
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.model, train_config.weight_decay),
            lr=train_config.peak_lr, betas=train_config.betas,
            eps=train_config.eps)

        # GradScaler is only meaningful on CUDA; on CPU it is disabled so the
        # same code runs for the local correctness gates.
        self.use_fp16 = (device.startswith("cuda")
                         and train_config.precision == "fp16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        self.train_stream = TokenStream(paths.split_bin("train"),
                                        model_config.context_len)
        self.val_stream = TokenStream(paths.split_bin("val"),
                                      model_config.context_len)

        self.step = 0
        self.rng = np.random.default_rng(train_config.seed)
        self.log_path = self.out_dir / "log.jsonl"
        self.interrupted = False

    # -- schedule ---------------------------------------------------------

    def set_lr(self, step: int) -> float:
        """Write the scheduled learning rate into the optimiser.

        The schedule is a pure function of ``step`` held on the config rather
        than a stateful ``LRScheduler``.  That is what makes resume exact: the
        LR after restarting is recomputed from the step number, so it cannot
        drift out of sync with the optimiser the way a separately-restored
        scheduler object can.

        Args:
            step: Current optimiser step.

        Returns:
            The learning rate applied.
        """
        lr = self.train_config.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean validation loss over a fixed number of batches.

        The generator is re-seeded identically on every call, so each
        evaluation scores the *same* validation batches.  Otherwise successive
        val losses would differ partly because the sample changed, and the
        curve would be noise on top of signal.

        Returns:
            Mean cross-entropy in nats.
        """
        self.model.eval()
        rng = np.random.default_rng(self.train_config.seed + 1)
        losses = []
        for _ in range(self.train_config.eval_batches):
            batch = self.val_stream.sample(self.train_config.micro_batch, rng)
            x, y = to_torch(batch, self.device)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.use_fp16):
                _, loss, _ = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    # -- checkpointing ----------------------------------------------------

    def save_checkpoint(self, path: Path, val_loss: float | None = None) -> Path:
        """Write a checkpoint holding everything needed to continue.

        Weights alone are not enough.  AdamW carries first- and second-moment
        estimates per parameter; dropping them restarts the optimiser cold and
        produces a visible loss spike.  The step number is needed because the
        LR schedule is a function of it, and the RNG states are needed so the
        data order and dropout masks continue rather than repeat.

        Args:
            path: Destination file.
            val_loss: Latest validation loss, recorded for bookkeeping.

        Returns:
            The path written.
        """
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "val_loss": val_loss,
            "arch": self.arch,
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "rng_state": {
                "numpy": self.rng.bit_generator.state,
                "torch": torch.get_rng_state(),
            },
        }
        # Write to a temporary file and rename: os.replace is atomic, so a
        # crash mid-write cannot leave a truncated checkpoint that would fail
        # to load and lose the whole run.
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: Path) -> dict:
        """Restore a run from a checkpoint.

        Args:
            path: Checkpoint file.

        Returns:
            The loaded payload.

        Raises:
            ValueError: If the checkpoint's architecture differs from this
                run's, which would silently load mismatched weights.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)

        saved = ModelConfig.from_dict(payload["model_config"])
        saved_arch = payload.get("arch", "gpt")
        if saved_arch != self.arch:
            raise ValueError(f"checkpoint is {saved_arch}, this run is {self.arch}")
        if saved != self.model_config:
            raise ValueError(
                f"checkpoint architecture does not match this run:\n"
                f"  checkpoint: {saved}\n  current:    {self.model_config}")

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = payload["step"]
        self.rng.bit_generator.state = payload["rng_state"]["numpy"]
        torch.set_rng_state(payload["rng_state"]["torch"].cpu()
                            if hasattr(payload["rng_state"]["torch"], "cpu")
                            else payload["rng_state"]["torch"])
        LOGGER.info("resumed from %s at step %d", path, self.step)
        return payload

    def maybe_resume(self) -> bool:
        """Continue from ``last.pt`` if it exists.

        Returns:
            True if a checkpoint was restored.
        """
        last = self.out_dir / "last.pt"
        if last.exists():
            self.load_checkpoint(last)
            return True
        return False

    # -- the loop ---------------------------------------------------------

    def log(self, record: dict) -> None:
        """Append one JSONL record to the run log.

        Args:
            record: Values to log.
        """
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def train_step(self) -> tuple[float, float]:
        """Run one optimiser step over ``grad_accum`` micro-batches.

        Returns:
            ``(loss, grad_norm, lr)`` for the step.
        """
        config = self.train_config
        lr = self.set_lr(self.step)

        # Zeroed once per optimiser step, not per micro-batch -- accumulating
        # into the .grad buffers across micro-batches is the whole mechanism.
        self.optimizer.zero_grad(set_to_none=True)

        total = 0.0
        for _ in range(config.grad_accum):
            batch = self.train_stream.sample(config.micro_batch, self.rng)
            x, y = to_torch(batch, self.device)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.use_fp16):
                _, loss, _ = self.model(x, y)
                loss = loss / config.grad_accum
            self.scaler.scale(loss).backward()
            total += loss.item()

        # Unscale BEFORE clipping so the norm compared against grad_clip is the
        # true gradient norm, not the fp16-scaled one.
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        return total, float(grad_norm), lr

    def run(self, max_steps: int | None = None) -> dict:
        """Train until ``max_steps``, checkpointing and evaluating as it goes.

        Args:
            max_steps: Override the config's step budget.

        Returns:
            Summary of the run.
        """
        config = self.train_config
        target = max_steps or config.max_steps

        # A SIGINT sets a flag rather than killing the process, so the loop can
        # finish the step it is on and checkpoint before exiting.  Losing an
        # hour of training to an impatient Ctrl-C is avoidable.
        def handle_interrupt(signum, frame):
            LOGGER.warning("interrupt received; checkpointing after this step")
            self.interrupted = True

        signal.signal(signal.SIGINT, handle_interrupt)

        self.model.train()
        started = time.time()
        tokens_per_step = config.tokens_per_step(self.model_config.context_len)
        best_val = float("inf")
        last_val = None

        LOGGER.info("%s: %d steps x %s tokens = %s tokens, peak_lr=%.2e",
                    config.language, target, f"{tokens_per_step:,}",
                    f"{target * tokens_per_step:,}", config.peak_lr)

        while self.step < target:
            step_started = time.time()
            loss, grad_norm, lr = self.train_step()
            elapsed = time.time() - step_started

            if self.step % 10 == 0 or self.step == 1:
                LOGGER.info(
                    "step %6d/%d  loss %7.4f  lr %.3e  |g| %6.3f  %6.0f tok/s",
                    self.step, target, loss, lr, grad_norm,
                    tokens_per_step / max(elapsed, 1e-9))

            record = {"step": self.step, "train_loss": loss, "lr": lr,
                      "grad_norm": grad_norm, "seconds": round(elapsed, 4),
                      "tokens": self.step * tokens_per_step}

            if self.step % config.eval_interval == 0 or self.step == target:
                last_val = self.evaluate()
                record["val_loss"] = last_val
                LOGGER.info("step %6d  VAL %7.4f  (ppl %.1f)",
                            self.step, last_val, math.exp(min(last_val, 20)))
                if last_val < best_val:
                    best_val = last_val
                    self.save_checkpoint(self.out_dir / "best.pt", last_val)

            self.log(record)

            if self.step % config.ckpt_interval == 0:
                self.save_checkpoint(self.out_dir / "last.pt", last_val)

            if self.interrupted:
                self.save_checkpoint(self.out_dir / "last.pt", last_val)
                LOGGER.warning("checkpointed at step %d; exiting", self.step)
                break

        self.save_checkpoint(self.out_dir / "last.pt", last_val)
        if last_val is None:
            last_val = self.evaluate()

        return {"language": config.language, "steps": self.step,
                "final_val_loss": last_val, "best_val_loss": best_val,
                "peak_lr": config.peak_lr,
                "tokens_seen": self.step * tokens_per_step,
                "wall_seconds": round(time.time() - started, 1),
                "interrupted": self.interrupted}


def main() -> None:
    """Pretrain one architecture on the shared Hindi recipe."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--arch", default="gpt",
                        help="Decoder architecture: gpt (llama from Stage 2)")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="Architecture config; defaults to the variant's")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    paths = Paths.for_variant(args.language, args.root, args.variant)
    model_config = ModelConfig.load(args.model_config or paths.model_config)
    train_config = TrainConfig.load(paths.train_config)

    overrides = {k: v for k, v in
                 (("peak_lr", args.peak_lr),
                                   ("warmup_steps", args.warmup_steps), ("max_steps", args.max_steps),
                  ("micro_batch", args.micro_batch),
                  ("grad_accum", args.grad_accum))
                 if v is not None}
    if overrides:
        train_config = TrainConfig.from_dict(
            {**train_config.to_dict(), **overrides})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s  params=%s", device,
                f"{model_config.parameter_count:,}")

    trainer = Trainer(model_config, train_config, paths, device, args.out,
                      arch=args.arch)
    if not args.no_resume:
        trainer.maybe_resume()

    summary = trainer.run(args.max_steps)
    LOGGER.info("done: %s", json.dumps(summary, indent=2))
    (trainer.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
