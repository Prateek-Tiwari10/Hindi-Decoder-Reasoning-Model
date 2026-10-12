"""Finetuning hyperparameters, serialised alongside every checkpoint.

Held as a dataclass rather than a dict so a typo in a field name fails at
construction instead of silently training with a default.  Every field is
written into the checkpoint, so a checkpoint found later states in full what
produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class SFTConfig:
    """One finetuning recipe.

    Attributes:
        language: Always ``"hindi"``; recorded so checkpoints round-trip.
        variant: Pretraining variant the base weights come from (``"16k"``).
        peak_lr: Peak learning rate after warmup.
        warmup_ratio: Warmup as a fraction of total steps.
        floor_ratio: Cosine floor, as a fraction of ``peak_lr``.
        epochs: Passes over the training split.
        micro_batch: Sequences per forward pass.
        grad_accum: Forward passes per optimizer step.  ``micro_batch *
            grad_accum`` is the effective batch in sequences.
        weight_decay: AdamW decay, applied to 2-D weights only.
        betas: AdamW betas.
        eps: AdamW epsilon.
        grad_clip: Global gradient-norm clip.
        precision: ``"fp16"`` or ``"fp32"``.
        seed: Master seed for init order, shuffling and dropout.
        eval_interval: Optimizer steps between validation reads.
        em_subset: Size of the fixed val subset used for exact match.
        em_max_new_tokens: Generation cap during val-EM.
        ckpt_interval: Optimizer steps between ``last.pt`` writes.
        length_bucket: Window size for length-bucketed shuffling; 0 disables.
        families: Restrict training to these task families, or None for all.
        run_name: Names the output directory.
    """

    language: str
    variant: str = "16k"
    peak_lr: float = 2e-4
    warmup_ratio: float = 0.03
    floor_ratio: float = 0.1
    epochs: int = 3
    micro_batch: int = 16
    grad_accum: int = 4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    precision: str = "fp16"
    seed: int = 1337
    eval_interval: int = 250
    em_subset: int = 500
    em_max_new_tokens: int = 160
    ckpt_interval: int = 250
    length_bucket: int = 512
    families: tuple[str, ...] | None = None
    run_name: str = "sft"
    #: Legacy record: the directory the course run read its splits from
    #: (``data_scaling_60k``).  Kept so old checkpoints' configs round-trip.
    data_dir: str = "data"
    #: Training split, relative to ``artifacts/finetune_data/``.  Empty means
    #: the legacy ``<data_dir>/hi/train.jsonl`` layout.
    train_file: str = ""
    #: Token-loss validation split (train-pool names), same convention.
    val_file: str = ""
    #: File used for exact-match validation and checkpoint selection, relative
    #: to ``artifacts/finetune_data/`` when ``train_file`` is set.  The
    #: performance build points this at ``val_heldout.jsonl``, whose names come
    #: from the HELD-OUT pool.  Selecting on the train-name validation set is
    #: what hid the memorization in Part 2, so this is deliberately separate
    #: from ``val.jsonl``, which is still scored for contrast.
    em_file: str = "val.jsonl"
    #: Stop after this many consecutive evaluations without a new best
    #: exact match.  Zero disables early stopping.
    early_stop_patience: int = 0
    #: Safety truncation.  Raised from 288 for the performance build: the
    #: larger name pool is built from compounds, which cost more subword
    #: tokens.  The longest Hindi training example is 206 tokens.
    max_len: int = 384

    @property
    def effective_batch(self) -> int:
        """Sequences per optimizer step."""
        return self.micro_batch * self.grad_accum

    def lr_at(self, step: int, total_steps: int) -> float:
        """Learning rate for one step: linear warmup, then cosine to a floor.

        The same shape as pretraining, and like pretraining it is a pure
        function of the step rather than a stateful scheduler object.  That is
        what makes resume exact -- the rate after restarting is recomputed from
        the step number and cannot drift out of sync with the optimizer.

        Args:
            step: Optimizer step, 0-based.
            total_steps: Total planned steps, used to place the cosine.

        Returns:
            The learning rate to apply.
        """
        import math

        warmup = max(1, int(self.warmup_ratio * total_steps))
        if step < warmup:
            return self.peak_lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (self.floor_ratio
                               + (1.0 - self.floor_ratio) * cosine)

    def to_dict(self) -> dict:
        """Serialise for the checkpoint payload and the run log."""
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        if self.families is not None:
            payload["families"] = list(self.families)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SFTConfig":
        """Rebuild from a serialised config, ignoring unknown keys.

        Args:
            payload: A dict produced by :meth:`to_dict`.

        Returns:
            The config.
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known}
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        if kwargs.get("families") is not None:
            kwargs["families"] = tuple(kwargs["families"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "SFTConfig":
        """Read a config from YAML or JSON.

        Args:
            path: ``.yaml`` or ``.json`` file.

        Returns:
            The config.
        """
        from model.config import read_config_file

        return cls.from_dict(read_config_file(path))

    def save(self, path: Path) -> Path:
        """Write this config to JSON.

        Args:
            path: Destination.

        Returns:
            The path written.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                              encoding="utf-8")
        return Path(path)
