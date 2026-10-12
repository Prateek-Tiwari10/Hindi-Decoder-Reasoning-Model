"""Supervised finetuning trainer for the Hindi reasoning task.

    python -m train.finetune --arch gpt --config configs/finetune_shared.yaml

One script for every architecture: ``--arch`` selects the decoder, ``--base``
the pretrained checkpoint it starts from.  Everything else -- data, order,
schedule, early stopping -- comes from ``configs/finetune_shared.yaml``.

WHAT IS REUSED AND WHAT IS NEW
------------------------------
The model is built by ``model.build_model``, imported, not reimplemented --
the finetune is the same forward pass and the same causal-LM objective, and the
architecture does not change.  Parameter grouping is Phase 2's too.  What is
new is the data pipeline (prompt-masked, dynamically padded) and this loop's
bookkeeping: exact-match validation, best-checkpoint selection, and epochs over
a finite dataset rather than random windows over a token stream.

CHECKPOINT FORMAT
-----------------
Identical key set to Phase 2's (``model``, ``optimizer``, ``scaler``, ``step``,
``val_loss``, ``model_config``, ``train_config``, ``rng_state``), so anything
that could load a pretraining checkpoint can load these.  Finetuning adds
``sft_config``, ``epoch`` and ``val_em`` alongside.  There is no separate
scheduler state to save because the schedule is a pure function of the step,
which is saved -- the same arrangement pretraining used.

TOKEN-WEIGHTED GRADIENT ACCUMULATION
------------------------------------
Each micro-batch's loss is the mean over ITS OWN completion tokens, and
different micro-batches contain different numbers of them.  Averaging those
means -- the usual ``loss / grad_accum`` -- would silently weight a batch of
short completions equally with a batch of long ones, so the optimizer would not
be descending the loss it reports.  Because the dataset is tokenized up front,
the exact token count of every micro-batch in a step is known BEFORE the
backward passes, so each micro-batch is scaled by its true share
``n_i / sum(n)`` and the accumulated gradient is exactly that of the
token-weighted mean.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from model.config import ModelConfig, Paths
from model import build_model
from train.pretrain import parameter_groups
from data import sft_data
from eval import completions as sft_generate
from train.finetune_config import SFTConfig

LOGGER = logging.getLogger("train.finetune")


def load_tokenizer(paths: Paths):
    """Open the language's frozen Phase 1 tokenizer, read-only.

    Args:
        paths: Resolved paths for this language and variant.

    Returns:
        A loaded ``SentencePieceProcessor``.

    Raises:
        FileNotFoundError: If the tokenizer is missing.
    """
    import sentencepiece as spm

    if not paths.tokenizer.exists():
        raise FileNotFoundError(f"tokenizer not found: {paths.tokenizer}")
    processor = spm.SentencePieceProcessor()
    processor.load(str(paths.tokenizer))
    return processor


def resolve_eos(tokenizer) -> int:
    """The end-of-sequence id, matching what pretraining used.

    ``common/export.py`` separated documents in the pretraining stream with
    ``processor.eos_id()``, falling back to the piece ``</s>``.  Finetuning
    reuses exactly that id so the stop token is one the base model has already
    seen many millions of times, rather than a token being given a new meaning.

    Args:
        tokenizer: Loaded SentencePiece processor.

    Returns:
        The EOS id.

    Raises:
        ValueError: If the model exposes no EOS at all, which would mean
            inventing a stop convention the base model never learned.
    """
    eos = tokenizer.eos_id()
    if eos < 0:
        eos = tokenizer.piece_to_id("</s>")
    if eos is None or eos < 0:
        raise ValueError(
            "tokenizer exposes no EOS id and no '</s>' piece; pretraining "
            "separated documents with one, so finetuning cannot proceed "
            "without matching it")
    return eos


def base_checkpoint(paths: Paths, override: Path | None = None) -> Path:
    """Locate the pretrained checkpoint a finetune starts from.

    Args:
        paths: Resolved paths for this variant.
        override: Explicit checkpoint (e.g. the LLaMA model's own final
            checkpoint).  Defaults to the variant's ``final_model``.

    Returns:
        Path to the base checkpoint.

    Raises:
        FileNotFoundError: If it is absent.  Guessing a different checkpoint
            would mean finetuning from the wrong recipe.
    """
    candidate = Path(override) if override else paths.final_model
    if not candidate.exists():
        raise FileNotFoundError(
            f"base checkpoint not found: {candidate}. The GPT baseline is "
            f"published at Prateek-Tiwari10/LMA_phase2 (16k/hindi/); place "
            f"checkpoint.pt at {paths.final_model}.")
    return candidate


class SFTTrainer:
    """Owns one finetuning run."""

    def __init__(self, sft_config: SFTConfig, paths: Paths, device: str,
                 out_dir: Path, data_root: Path, arch: str = "gpt",
                 base: Path | None = None) -> None:
        """Set up the run: base weights, optimizer, data, schedule.

        Args:
            sft_config: The recipe.
            paths: Resolved paths for this variant.
            device: Torch device.
            out_dir: Where checkpoints, the log and the config are written.
            data_root: ``artifacts/finetune_data``; split files are resolved
                against it.
            arch: Decoder architecture of the base checkpoint.
            base: Explicit pretrained checkpoint; defaults to the variant's.

        Raises:
            ValueError: If the base checkpoint's architecture is not the
                carried-forward recipe, or is for a different language.
        """
        self.config = sft_config
        self.paths = paths
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root)

        torch.manual_seed(sft_config.seed)
        np.random.seed(sft_config.seed)

        # -- base weights -------------------------------------------------
        self.arch = arch
        self.base_path = base_checkpoint(paths, base)
        payload = torch.load(self.base_path, map_location="cpu",
                             weights_only=False)
        self.model_config = ModelConfig.from_dict(payload["model_config"])

        if self.model_config.language != paths.language:
            raise ValueError(
                f"{self.base_path} is a {self.model_config.language} model but "
                f"this run is for {paths.language}; one language must never be "
                f"initialised from the other's weights")
        if self.model_config.vocab_size != paths.vocab_size:
            raise ValueError(
                f"{self.base_path} has vocab_size "
                f"{self.model_config.vocab_size}, expected {paths.vocab_size} "
                f"for variant {paths.variant!r}")

        saved_arch = payload.get("arch", "gpt")
        if saved_arch != arch:
            raise ValueError(f"{self.base_path} is a {saved_arch} checkpoint "
                             f"but --arch {arch} was requested")
        self.model = build_model(self.model_config, device, arch=arch)
        self.model.load_state_dict(payload["model"])
        self.base_step = int(payload.get("step", -1))
        self.base_val_loss = payload.get("val_loss")
        LOGGER.info("loaded base %s (step %d, val_loss %.4f)",
                    self.base_path, self.base_step, self.base_val_loss or 0.0)

        # A FRESH optimizer, deliberately.  The pretraining moments describe a
        # different objective at a much larger learning rate; carrying them in
        # would apply stale second-moment estimates to the first SFT steps.
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.model, sft_config.weight_decay),
            lr=sft_config.peak_lr, betas=tuple(sft_config.betas),
            eps=sft_config.eps)

        self.use_fp16 = (device.startswith("cuda")
                         and sft_config.precision == "fp16")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        # -- data ---------------------------------------------------------
        self.tokenizer = load_tokenizer(paths)
        self.eos_id = resolve_eos(self.tokenizer)
        # Explicit split files (relative to the finetune-data root) when the
        # config names them; otherwise the legacy <data_dir>/hi/ layout.
        if sft_config.train_file:
            lang_dir = Path(data_root)
            train_path = lang_dir / sft_config.train_file
            val_path = lang_dir / sft_config.val_file
        else:
            lang_dir = Path(data_root) / sft_config.data_dir / "hi"
            train_path = lang_dir / "train.jsonl"
            val_path = lang_dir / "val.jsonl"

        self.train_examples, self.train_stats = sft_data.load_examples(
            train_path, self.tokenizer, self.eos_id,
            max_len=sft_config.max_len, families=sft_config.families)
        self.val_examples, self.val_stats = sft_data.load_examples(
            val_path, self.tokenizer, self.eos_id,
            max_len=sft_config.max_len)

        # Exact match is scored on `em_file`, which may be a DIFFERENT split
        # from the token-loss validation set.  The performance build points it
        # at held-out names, because selecting on train-name validation is what
        # let a memorizing model look finished at 99.8%.
        em_path = lang_dir / sft_config.em_file
        if em_path == val_path:
            em_pool, self.em_stats = self.val_examples, self.val_stats
        else:
            em_pool, self.em_stats = sft_data.load_examples(
                em_path, self.tokenizer, self.eos_id,
                max_len=sft_config.max_len)
        self.em_file = str(em_path)

        # A FIXED subset, drawn once with its own generator, so every val-EM
        # read scores the same examples and the curve is signal rather than a
        # changing sample.
        subset_rng = np.random.default_rng(sft_config.seed + 7)
        subset = subset_rng.permutation(len(em_pool))
        subset = subset[:min(sft_config.em_subset, len(em_pool))]
        self.em_examples = [em_pool[i] for i in sorted(subset)]

        steps_per_epoch = max(
            1, len(self.train_examples) // self.config.effective_batch)
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * sft_config.epochs

        self.step = 0
        self.epoch = 0
        self.best_em = -1.0
        self.best_step = -1
        self.evals_since_best = 0
        self.early_stopped = False
        self.rng = np.random.default_rng(sft_config.seed)
        self.log_path = self.out_dir / "log.jsonl"
        self.interrupted = False
        self.truncation_warned = False

        self.language_config = self._load_language_config()

    def _load_language_config(self) -> dict:
        """Load the Part 1 surface config, for the answer cue used by val-EM.

        Returns:
            The parsed language config.
        """
        from data.reasoning.generator import load_language_config

        # hi.yaml and hi_perf.yaml share every surface string (cues, labels);
        # the course run read hi.yaml here, so this does too.
        return load_language_config(self.paths.reasoning_dir / "configs" / "hi.yaml")

    # -- schedule ---------------------------------------------------------

    def set_lr(self, step: int) -> float:
        """Write the scheduled learning rate into the optimizer.

        Args:
            step: Optimizer step.

        Returns:
            The learning rate applied.
        """
        lr = self.config.lr_at(step, self.total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- validation -------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        """Token loss over the whole validation split.

        Reported as the token-weighted mean over completion tokens, which is
        the same quantity the training loss reports -- so the two curves are
        directly comparable rather than differing by a batching artefact.

        Returns:
            ``{"val_loss": float, "val_tokens": int}``.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        order = np.arange(len(self.val_examples))
        for start in range(0, len(order), self.config.micro_batch):
            batch = [self.val_examples[i]
                     for i in order[start:start + self.config.micro_batch]]
            x, y, n_supervised = sft_data.collate(batch, self.eos_id,
                                                  self.device)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.use_fp16):
                _, loss, _ = self.model(x, y)
            total_loss += float(loss.detach()) * n_supervised
            total_tokens += n_supervised
        self.model.train()
        return {"val_loss": total_loss / max(1, total_tokens),
                "val_tokens": total_tokens}

    def validate_em(self) -> dict:
        """Exact match on the fixed validation subset.

        Returns:
            The summary from :func:`eval.completions.exact_match`, without
            the per-example predictions.
        """
        result = sft_generate.exact_match(
            self.model, self.em_examples, self.tokenizer,
            self.language_config, self.eos_id, self.device,
            max_new_tokens=self.config.em_max_new_tokens,
            max_batch=self.config.micro_batch * 2)
        result.pop("predictions", None)
        return result

    # -- checkpointing ----------------------------------------------------

    def save_checkpoint(self, path: Path, val_loss: float | None = None,
                        val_em: float | None = None) -> Path:
        """Write a resume-capable checkpoint in Phase 2's format.

        Args:
            path: Destination.
            val_loss: Latest validation token loss.
            val_em: Latest validation exact match.

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
            "train_config": self.config.to_dict(),
            "rng_state": {
                "numpy": self.rng.bit_generator.state,
                "torch": torch.get_rng_state(),
                # CUDA's generator is SEPARATE from the CPU one, and dropout on
                # a CUDA tensor draws from it.  Phase 2's format saved only the
                # CPU state, which is complete on CPU and silently incomplete
                # on GPU: a resumed run would draw different dropout masks and
                # its loss curve would diverge from the pre-kill curve by
                # roughly 1e-2.  Gate P4 caught exactly that.  Additive key, so
                # Phase 2's loader (which reads "numpy" and "torch") is
                # unaffected.
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
            # Finetuning-specific, additive to the Phase 2 key set.
            "sft_config": self.config.to_dict(),
            "epoch": self.epoch,
            "val_em": val_em,
            "base_checkpoint": str(self.base_path),
            "best_em": self.best_em,
            "best_step": self.best_step,
        }
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
            ValueError: If the architecture differs from this run's.
        """
        payload = torch.load(path, map_location=self.device,
                             weights_only=False)
        saved = ModelConfig.from_dict(payload["model_config"])
        if saved != self.model_config:
            raise ValueError(
                f"checkpoint architecture does not match this run:\n"
                f"  checkpoint: {saved}\n  current:    {self.model_config}")

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = payload["step"]
        self.epoch = payload.get("epoch", 0)
        self.best_em = payload.get("best_em", -1.0)
        self.best_step = payload.get("best_step", -1)
        self.rng.bit_generator.state = payload["rng_state"]["numpy"]
        torch_state = payload["rng_state"]["torch"]
        torch.set_rng_state(torch_state.cpu() if hasattr(torch_state, "cpu")
                            else torch_state)
        cuda_state = payload["rng_state"].get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            # Guarded on device count: a checkpoint written on a multi-GPU box
            # carries one state per device and set_rng_state_all would raise
            # here, which is a worse failure than continuing without it.
            if len(cuda_state) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(
                    [s.cpu() if hasattr(s, "cpu") else s for s in cuda_state])
            else:
                LOGGER.warning(
                    "checkpoint has %d CUDA RNG states but this host has %d "
                    "device(s); dropout will not resume bit-exactly",
                    len(cuda_state), torch.cuda.device_count())
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

    def train_step(self, batches: list[list]) -> tuple[float, float, int]:
        """One optimizer step over pre-collected micro-batches.

        Args:
            batches: ``grad_accum`` micro-batches of examples.

        Returns:
            ``(loss, grad_norm, supervised_tokens)`` where the loss is the
            token-weighted mean over the whole step.
        """
        collated = [sft_data.collate(b, self.eos_id, self.device)
                    for b in batches]
        total_tokens = sum(n for _, _, n in collated)

        self.optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for x, y, n_supervised in collated:
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.use_fp16):
                _, loss, _ = self.model(x, y)
            # Exact token weighting: this micro-batch's share of the step.
            share = n_supervised / max(1, total_tokens)
            self.scaler.scale(loss * share).backward()
            step_loss += float(loss.detach()) * share

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return step_loss, float(grad_norm), total_tokens

    def run(self, max_steps: int | None = None) -> dict:
        """Finetune, validating and checkpointing as configured.

        Args:
            max_steps: Stop after this many optimizer steps.  Used by the LR
                mini-sweep to run a fraction of an epoch.

        Returns:
            A summary of the run.
        """
        def on_interrupt(signum, frame):
            LOGGER.warning("interrupt received; finishing the current step")
            self.interrupted = True

        previous = signal.signal(signal.SIGINT, on_interrupt)
        limit = min(self.total_steps, max_steps or self.total_steps)
        started = time.time()
        self.model.train()

        while self.step < limit and not self.interrupted and not self.early_stopped:
            epoch = self.step // self.steps_per_epoch
            self.epoch = epoch
            # Rebuilt from (seed, epoch), so a resume reconstructs the very
            # same order and skips exactly what it already consumed.
            micro_batches = sft_data.epoch_batches(
                self.train_examples, self.config.micro_batch,
                self.config.seed, epoch, self.config.length_bucket)
            steps_done_this_epoch = self.step % self.steps_per_epoch
            skip = steps_done_this_epoch * self.config.grad_accum

            pending: list[list] = []
            for position, micro in enumerate(micro_batches):
                if position < skip:
                    continue
                pending.append(micro)
                if len(pending) < self.config.grad_accum:
                    continue

                lr = self.set_lr(self.step)
                loss, grad_norm, tokens = self.train_step(pending)
                pending = []
                self.step += 1
                steps_done_this_epoch += 1

                self.log({"step": self.step, "epoch": epoch,
                          "train_loss": loss, "lr": lr,
                          "grad_norm": grad_norm, "tokens": tokens,
                          "elapsed": time.time() - started})

                if (self.step % self.config.eval_interval == 0
                        or self.step == limit):
                    val = self.validate()
                    em = self.validate_em()
                    self.log({"step": self.step, "epoch": epoch,
                              "val_loss": val["val_loss"],
                              "val_em": em["exact_match"],
                              "val_em_correct": em["correct"],
                              "val_em_total": em["total"],
                              "val_no_cue": em["no_answer_cue"],
                              "val_hit_cap": em["hit_length_cap"],
                              "elapsed": time.time() - started})
                    LOGGER.info(
                        "step %d/%d loss %.4f val %.4f EM %.3f lr %.2e",
                        self.step, limit, loss, val["val_loss"],
                        em["exact_match"], lr)

                    if em["exact_match"] > self.best_em:
                        self.best_em = em["exact_match"]
                        self.best_step = self.step
                        self.evals_since_best = 0
                        self.save_checkpoint(self.out_dir / "best.pt",
                                             val["val_loss"],
                                             em["exact_match"])
                    else:
                        self.evals_since_best += 1
                        patience = self.config.early_stop_patience
                        if patience and self.evals_since_best >= patience:
                            LOGGER.info(
                                "early stop: %d evaluations without a new "
                                "best exact match (best %.4f at step %d)",
                                self.evals_since_best, self.best_em,
                                self.best_step)
                            self.early_stopped = True

                if self.step % self.config.ckpt_interval == 0:
                    self.save_checkpoint(self.out_dir / "last.pt")

                if (self.step >= limit or self.interrupted
                        or self.early_stopped
                        or steps_done_this_epoch >= self.steps_per_epoch):
                    break

        self.save_checkpoint(self.out_dir / "last.pt")
        signal.signal(signal.SIGINT, previous)

        elapsed = time.time() - started
        summary = {
            "language": self.paths.language,
            "run_name": self.config.run_name,
            "base_checkpoint": str(self.base_path),
            "base_step": self.base_step,
            "base_val_loss": self.base_val_loss,
            "steps_completed": self.step,
            "total_steps_planned": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.config.epochs,
            "best_val_em": self.best_em,
            "best_step": self.best_step,
            "interrupted": self.interrupted,
            "early_stopped": self.early_stopped,
            "em_file": self.em_file,
            "em_stats": self.em_stats,
            "wall_clock_seconds": round(elapsed, 1),
            "gpu_hours": round(elapsed / 3600.0, 4),
            "device": self.device,
            "train_stats": self.train_stats,
            "val_stats": self.val_stats,
            "em_subset_size": len(self.em_examples),
            "eos_id": self.eos_id,
            "sft_config": self.config.to_dict(),
            "arch": self.arch,
            "model_config": self.model_config.to_dict(),
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--arch", default="gpt",
                        help="Decoder architecture: gpt (llama from Stage 2)")
    parser.add_argument("--variant", default="16k",
                        help="Pretraining variant whose tokenizer/paths to use")
    parser.add_argument("--base", type=Path, default=None,
                        help="Pretrained checkpoint to start from; defaults "
                             "to the variant's final checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/finetune_shared.yaml"),
                        help="SFT config (YAML or JSON)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--relation-only", action="store_true",
                        help="train on task_family == 'relation' only")
    parser.add_argument("--em-file", default=None,
                        help="file scored for exact match, relative to "
                             "artifacts/finetune_data/")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = (SFTConfig.load(args.config) if args.config
              else SFTConfig(language=args.language))
    config.language = args.language
    config.variant = args.variant
    for attr, value in (("peak_lr", args.peak_lr), ("epochs", args.epochs),
                        ("micro_batch", args.micro_batch),
                        ("grad_accum", args.grad_accum),
                        ("run_name", args.run_name),
                        ("em_file", args.em_file),
                        ("early_stop_patience", args.early_stop_patience),
                        ("eval_interval", args.eval_interval),
                        ("ckpt_interval", args.ckpt_interval)):
        if value is not None:
            setattr(config, attr, value)
    if args.relation_only:
        config.families = ("relation",)
        config.run_name = config.run_name or "sft"
        if not config.run_name.endswith("_relonly"):
            config.run_name = f"{config.run_name}_relonly"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.root.resolve()
    paths = Paths.for_variant(args.language, root, args.variant)
    out_dir = args.out or (paths.artifacts / "runs"
                           / f"finetune_{args.arch}_{config.run_name}")

    trainer = SFTTrainer(config, paths, device, out_dir, paths.finetune_data,
                         arch=args.arch, base=args.base)
    config.save(out_dir / "config.json")
    if not args.no_resume:
        trainer.maybe_resume()

    summary = trainer.run(max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("train_stats", "val_stats", "sft_config",
                                   "model_config")}, indent=2))


if __name__ == "__main__":
    main()
