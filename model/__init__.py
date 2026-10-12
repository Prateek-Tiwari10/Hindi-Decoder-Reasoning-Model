"""Decoder architectures for the Hindi GPT-vs-LLaMA study.

``build_model(config, device, arch)`` is the single entry point every training
and evaluation script uses, so the two architectures go through identical code
everywhere except the model itself.
"""

from __future__ import annotations

from model.config import ModelConfig

#: Architectures this package can build.  ``"llama"`` is added in Stage 2.
ARCHITECTURES = ("gpt",)


def build_model(config: ModelConfig, device: str = "cpu", arch: str = "gpt",
                verify_budget: bool = True):
    """Construct a decoder of the requested architecture.

    Args:
        config: Architecture configuration.
        device: Where to place the model.
        arch: One of :data:`ARCHITECTURES`.
        verify_budget: Assert the live parameter count matches the analytic one
            and sits inside the 25-30M budget.

    Returns:
        The model on ``device``.

    Raises:
        ValueError: If ``arch`` is unknown.
    """
    if arch == "gpt":
        from model.gpt import build_model as build_gpt

        return build_gpt(config, device, verify_budget=verify_budget)
    raise ValueError(f"unknown arch {arch!r}; known: {ARCHITECTURES}")
