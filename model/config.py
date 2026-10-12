"""Configuration objects for the Hindi decoder-only language models.

ONE RECIPE, SEVERAL ARCHITECTURES
---------------------------------
Every model in this repository is a Hindi model trained on the same 16K-BPE
token bins with the same recipe.  What varies is the *architecture* (the GPT-
style baseline, the LLaMA-style comparison, the no-positional ablation).  The
data, schedule, batch and seed live in :class:`TrainConfig` and are shared, so
any difference between two runs is a fact about the architecture rather than
about the setup.

WHY THE ARCHITECTURAL CHOICES ARE FIELDS AND NOT CONSTANTS
----------------------------------------------------------
``pre_norm``, ``tie_weights`` and the rest are recorded as fields even though
nothing ever varies them.  Every checkpoint serialises its config, so a
checkpoint found on disk months later states in full what produced it instead of
implying it by which version of the code was checked out.

THE PARAMETER BUDGET
--------------------
The models are held at 25-30M parameters.  ``ModelConfig.parameter_count``
computes that analytically from the shapes alone, so the budget can be checked
before PyTorch is even imported.  ``model/gpt.py`` counts the real tensors and
the two numbers are asserted equal -- an analytic count that agrees with a live
count means the model that got built is the model that was designed.

CONFIG FILES
------------
Configs are read from YAML (``configs/*.yaml``) or JSON (checkpoint sidecars and
older run directories); :func:`read_config_file` dispatches on the suffix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: The parameter budget, in parameters.
PARAM_BUDGET = (25_000_000, 30_000_000)

#: The architecture variants, mapping the tag that appears in file names to the
#: vocabulary its token bins were built with.  A variant is a *config*, not new
#: model code.
#:
#:   ``"16k"``        the GPT-style baseline -- 16K vocab, d_model 384, 11 layers
#:   ``"16k_nopos"``  the ablation -- identical to ``"16k"`` except that
#:                    ``positional`` is ``"none"``, so the model has no
#:                    positional table at all.
VARIANTS: dict[str, int] = {
    "16k": 16_000,
    "16k_nopos": 16_000,
}

#: The variant every script uses unless told otherwise.
DEFAULT_VARIANT = "16k"

#: Gitignored home of every large binary: token bins, checkpoints, the
#: finetuning corpus.  Nothing under it is committed; README links say where
#: each file is published.
ARTIFACTS = "artifacts"


def read_config_file(path: Path) -> dict[str, Any]:
    """Parse a YAML or JSON config file into a plain mapping.

    Args:
        path: ``.yaml``/``.yml`` or ``.json`` file.

    Returns:
        The parsed mapping.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _filter_known(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that are not fields of ``cls``.

    Serialised configs carry derived, human-readable extras (``d_k``,
    ``parameter_count``) so a config file can be read without running any code.
    Those are outputs, not inputs, and must not be fed back into the
    constructor.

    Args:
        cls: Dataclass being reconstructed.
        payload: Parsed JSON object.

    Returns:
        Only the entries naming a real field.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in payload.items() if k in known}


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of one decoder-only Transformer.

    Frozen because the architecture is fixed before training starts and a
    checkpoint's config must describe the weights it sits next to.  Anything
    that changes between runs belongs in :class:`TrainConfig`.
    """

    #: Which model this is.  The ONLY field that differs between H and L.
    language: str = "hindi"

    vocab_size: int = 24_000
    d_model: int = 448
    n_layers: int = 7
    n_heads: int = 8
    d_ff: int = 1792
    #: Also the size of the learned positional table, so it is a hard cap on
    #: sequence length -- position 512 has no embedding row and will raise.
    context_len: int = 512
    dropout: float = 0.1

    # Locked architectural decisions, recorded rather than assumed.
    positional: str = "learned_absolute"
    pre_norm: bool = True
    activation: str = "gelu"
    tie_weights: bool = True
    #: LayerNorm epsilon; PyTorch's default, pinned so it is in the checkpoint.
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        """Reject any configuration the model code could not honour."""
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads="
                f"{self.n_heads}; the heads would not tile the channel axis")
        if self.vocab_size > 65_535:
            raise ValueError(
                f"vocab_size={self.vocab_size} does not fit uint16, but the "
                "Phase 1 .bin exports are uint16")
        if self.positional not in ("learned_absolute", "none"):
            raise ValueError(
                f"positional={self.positional!r} is not implemented; use "
                '"learned_absolute" or "none" (the ablation)')

    @property
    def d_k(self) -> int:
        """Per-head channel width -- the ``d_k`` in the ``1/sqrt(d_k)`` scaling."""
        return self.d_model // self.n_heads

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters, computed from shapes alone.

        Counts the token embedding once: with weight tying the output head *is*
        the embedding matrix viewed transposed, so counting it twice would
        overstate the model by a whole ``vocab_size x d_model`` block.

        Returns:
            Number of trainable scalars the model will hold.
        """
        d, ff = self.d_model, self.d_ff

        token_embedding = self.vocab_size * d       # tied: counted once
        # The ablation removes the table entirely rather than zeroing it, so it
        # contributes no parameters and receives no gradient.
        positions = 0 if self.positional == "none" else self.context_len * d

        # Q, K, V and the output projection: four (d, d) matrices with biases.
        attention = 4 * (d * d + d)
        # Two LayerNorms per block, each a weight and a bias of width d.
        norms = 2 * (2 * d)
        feedforward = (d * ff + ff) + (ff * d + d)
        per_block = attention + norms + feedforward

        final_norm = 2 * d
        # The output head is tied and carries no bias, so it adds nothing.
        return (token_embedding + positions
                + self.n_layers * per_block + final_norm)

    @property
    def within_budget(self) -> bool:
        """Whether the parameter count sits inside the brief's 25-30M window."""
        low, high = PARAM_BUDGET
        return low <= self.parameter_count <= high

    def to_dict(self) -> dict[str, Any]:
        """Serialise, appending derived values for the benefit of human readers.

        Returns:
            JSON-ready mapping of every field plus ``d_k`` and
            ``parameter_count``.
        """
        payload = asdict(self)
        payload["d_k"] = self.d_k
        payload["parameter_count"] = self.parameter_count
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        """Rebuild from a serialised mapping, ignoring derived extras.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed config.
        """
        return cls(**_filter_known(cls, payload))

    def save(self, path: Path) -> Path:
        """Write the config to JSON.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        """Read a config back from JSON.

        Args:
            path: File written by :meth:`save`.

        Returns:
            The reconstructed config.
        """
        return cls.from_dict(read_config_file(path))


@dataclass(frozen=True)
class TrainConfig:
    """The training recipe.

    Every field except ``peak_lr`` is frozen by the plan: the learning-rate
    sweep varies that one value and nothing else, which is what lets the
    sweep's winner transfer to the production run.

    The effective batch is ``micro_batch * grad_accum * context_len`` tokens.
    It is held at 32,768 in both the sweep and the production run; if a GPU
    cannot fit ``micro_batch=32``, halve it and double ``grad_accum`` so the
    product -- and therefore the optimisation problem -- is unchanged.
    """

    language: str = "hindi"

    # --- the swept knob ---
    peak_lr: float = 6e-4

    # --- schedule ---
    warmup_steps: int = 750
    max_steps: int = 15_000
    #: Cosine decays to ``floor_ratio * peak_lr`` rather than to zero, so the
    #: last steps still make progress instead of freezing the weights.
    floor_ratio: float = 0.1

    # --- batch ---
    micro_batch: int = 32
    grad_accum: int = 2

    # --- AdamW ---
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # --- bookkeeping ---
    eval_interval: int = 250
    #: Validation batches averaged per evaluation.  Fixed so the val loss is a
    #: comparable number across steps and across runs.
    eval_batches: int = 50
    ckpt_interval: int = 500
    seed: int = 1337
    #: fp16 + GradScaler, not bf16: Kaggle's T4 and P100 are pre-Ampere.
    precision: str = "fp16"
    #: Free-text tag distinguishing e.g. sweep runs from the production run.
    run_name: str = "production"

    def __post_init__(self) -> None:
        """Reject recipes the training loop could not honour."""
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps={self.warmup_steps} must be below "
                f"max_steps={self.max_steps}, or the LR never decays")
        if not 0.0 <= self.floor_ratio <= 1.0:
            raise ValueError("floor_ratio must be a fraction of the peak LR")
        if self.micro_batch < 1 or self.grad_accum < 1:
            raise ValueError("micro_batch and grad_accum must be positive")

    def tokens_per_step(self, context_len: int) -> int:
        """Tokens consumed by one optimiser step.

        Args:
            context_len: Sequence length from the :class:`ModelConfig`.

        Returns:
            ``micro_batch * grad_accum * context_len``.
        """
        return self.micro_batch * self.grad_accum * context_len

    def total_tokens(self, context_len: int) -> int:
        """Tokens the whole run will consume.

        Args:
            context_len: Sequence length from the :class:`ModelConfig`.

        Returns:
            Token budget of the run.
        """
        return self.tokens_per_step(context_len) * self.max_steps

    def lr_at(self, step: int) -> float:
        """Learning rate at ``step`` under linear warmup then cosine decay.

        Kept here rather than in the training loop so the schedule can be
        plotted and tested without constructing an optimiser, and so the
        checkpoint's config alone determines the LR at any step on resume.

        Args:
            step: Optimiser step, zero-based.

        Returns:
            The learning rate to use for that step.
        """
        import math

        floor = self.floor_ratio * self.peak_lr
        if step < self.warmup_steps:
            # Linear from 0 to peak.  Step 0 gets a non-zero LR (hence the +1)
            # so the very first update is not a no-op.
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return floor
        progress = ((step - self.warmup_steps)
                    / (self.max_steps - self.warmup_steps))
        return floor + 0.5 * (self.peak_lr - floor) * (
            1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict[str, Any]:
        """Serialise every field.

        Returns:
            JSON-ready mapping.  ``betas`` becomes a list, as JSON has no
            tuples; :meth:`from_dict` restores it.
        """
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        """Rebuild from a serialised mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed config, with ``betas`` back as a tuple.
        """
        known = _filter_known(cls, payload)
        if "betas" in known:
            known["betas"] = tuple(known["betas"])
        return cls(**known)

    def save(self, path: Path) -> Path:
        """Write the config to JSON.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        """Read a config back from JSON.

        Args:
            path: File written by :meth:`save`.

        Returns:
            The reconstructed config.
        """
        return cls.from_dict(read_config_file(path))


@dataclass(frozen=True)
class Paths:
    """Where the Hindi data, configs and artifacts live.

    Centralised so no module hard-codes a path.  Code and small configs are
    committed; every large binary sits under :data:`ARTIFACTS`, which is
    gitignored.

    ``variant`` selects the architecture config and its checkpoints.  Variants
    that share a tokenizer share their token bins.
    """

    language: str = "hindi"
    root: Path = field(default_factory=Path)
    #: Tag naming the architecture variant; a key of :data:`VARIANTS`.
    variant: str = DEFAULT_VARIANT
    #: Vocabulary of the tokenizer whose bins this variant reads.  Held here
    #: rather than read off the :class:`ModelConfig` because the data paths are
    #: needed before any config is loaded.
    vocab_size: int = VARIANTS[DEFAULT_VARIANT]

    @classmethod
    def for_variant(cls, language: str = "hindi", root: Path = Path("."),
                    variant: str = DEFAULT_VARIANT) -> "Paths":
        """Build the paths for one variant, taking its vocabulary from the registry.

        Using the registry rather than a caller-supplied vocabulary is what
        stops a run from pairing a config with the wrong token bins -- a
        mismatch that would not crash, it would simply train on text the
        tokenizer never wrote.

        Args:
            language: Must be ``"hindi"``; kept as an argument so configs and
                checkpoints that record it still round-trip.
            root: Repository root.
            variant: A key of :data:`VARIANTS`.

        Returns:
            The paths for that variant.

        Raises:
            ValueError: If ``language`` is not Hindi or ``variant`` is unknown.
        """
        if language != "hindi":
            raise ValueError(f"this repository is Hindi-only, got {language!r}")
        if variant not in VARIANTS:
            known = ", ".join(repr(v) for v in VARIANTS)
            raise ValueError(f"unknown variant {variant!r}; known: {known}")
        return cls(language, Path(root), variant, VARIANTS[variant])

    @property
    def artifacts(self) -> Path:
        """Root of the gitignored artifact tree."""
        return self.root / ARTIFACTS

    @property
    def suffix(self) -> str:
        """``"_16k"`` / ``"_16k_nopos"``: appended to per-variant file names."""
        return f"_{self.variant}"

    @property
    def bin_suffix(self) -> str:
        """``"_16k"`` for the 16K bins.

        Keyed on the vocabulary rather than the variant tag: two variants that
        share a tokenizer share their bins.
        """
        return f"_{self.vocab_size // 1000}k"

    @property
    def tokenizer(self) -> Path:
        """The SentencePiece BPE model this variant's bins were built with."""
        return self.root / "tokenizer" / f"hindi_bpe_{self.vocab_size}.model"

    @property
    def bin_dir(self) -> Path:
        """Directory holding the uint16 token streams."""
        return self.artifacts / "data" / "bin"

    def split_bin(self, split: str) -> Path:
        """Token stream for one split.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            Path to the ``uint16`` memmap.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}")
        return self.bin_dir / f"{split}{self.bin_suffix}.bin"

    @property
    def model_config(self) -> Path:
        """Serialised :class:`ModelConfig` for this variant."""
        if self.variant == "16k":
            return self.root / "configs" / "gpt_25m.yaml"
        return self.root / "ablation" / "configs" / f"model{self.suffix}.json"

    @property
    def train_config(self) -> Path:
        """Serialised :class:`TrainConfig` for this variant."""
        if self.variant == "16k":
            return self.root / "configs" / "pretrain_shared.yaml"
        return self.root / "ablation" / "configs" / f"train{self.suffix}.json"

    @property
    def ckpt_dir(self) -> Path:
        """Where training checkpoints of a new run are written.

        A subdirectory per variant, so a run can never resume from another
        architecture's ``last.pt``.
        """
        return self.artifacts / "runs" / f"pretrain{self.suffix}"

    @property
    def eval_dir(self) -> Path:
        """Where evaluation outputs are written."""
        return self.root / "reports" / "eval"

    @property
    def final_model(self) -> Path:
        """The finished pretrained checkpoint for this variant."""
        if self.variant == "16k":
            return self.artifacts / "checkpoints" / "gpt" / "final_16k.pt"
        return self.artifacts / "checkpoints" / "ablation" / "ablated_hindi.pt"

    # -- finetuning (reasoning task) ---------------------------------------

    @property
    def reasoning_dir(self) -> Path:
        """Committed generator code, configs, name pools and manifest."""
        return self.root / "data" / "reasoning"

    @property
    def reasoning_config(self) -> Path:
        """Hindi surface config of the reasoning generator (large name pools)."""
        return self.reasoning_dir / "configs" / "hi_perf.yaml"

    @property
    def finetune_data(self) -> Path:
        """Generated reasoning corpus: ``train/``, ``validation/``, ``test/``."""
        return self.artifacts / "finetune_data"

    @property
    def gpt_finetuned(self) -> Path:
        """The fixed GPT-ft baseline (``ft_60k`` ``best.pt``)."""
        return self.artifacts / "checkpoints" / "gpt_ft" / "ft_60k" / "best.pt"

    @property
    def reports(self) -> Path:
        """Generated results: plots, tables, metrics JSON, RESULTS files."""
        return self.root / "reports"
