"""Ablation -- where does the no-positional model rebuild position?

THE QUESTION
------------
:mod:`ablation.order_probe` establishes *that* the 11-layer ablated model is
order sensitive despite having no positional embedding table.  It does not say
*how*.  This module localises the mechanism by asking, layer by layer, how much
of a token's absolute position can be read linearly off the residual stream.

The method is a linear decoder.  Take the residual stream ``h[b, t, :]`` after
each layer, fit ridge regression from ``h`` to the position index ``t`` on one
set of sequences, and score it on held-out sequences.  Two numbers come out:
``R^2`` (fraction of positional variance recovered) and mean absolute error in
tokens.  Fitting on one set of sequences and scoring on another matters because
``C = 384`` free parameters against 128 distinct position values would otherwise
be free to memorise the training draw.

WHAT EACH MODEL SHOULD SHOW IF THE STORY IN :mod:`ablation.order_probe` IS RIGHT
-----------------------------------------------------------------------------
The **standard** model is handed position at the input: ``embed(token) +
embed(position)``.  Position should therefore be linearly decodable from the
very first tap, before any attention has run, at close to ceiling.

The **ablated** model gets no such input.  Its embedding output is a pure
function of the token id, so at the first tap position must be decodable only
to the extent that *which token appears* correlates with *where* -- sentence
boundaries, document starts, and so on.  That floor is not zero but it is low.
If the causal mask is the route by which order returns, decodability must then
*climb* across the early layers, and it should climb fastest where the
attention statistics say the interesting thing happens.

The probe is directional, not merely descriptive: it was written down before
the numbers were read, and either model could have falsified it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from model.config import Paths
from data.token_stream import TokenStream
from eval.lm_metrics import load_trained
from ablation.order_probe import PROBE_SEED, sample_windows

LOGGER = logging.getLogger("ablation.position_decoding")

N_SEQUENCES = 192
SEQ_LEN = 128
BATCH = 16
#: Ridge penalty.  Large enough to keep the 384x384 normal equations
#: well-conditioned, small enough not to bias a fit with ~12k rows per layer.
RIDGE_LAMBDA = 1.0
#: Position 0 is dropped: every model can identify it from the causal mask
#: alone (it is the only query with one visible key), so including it would
#: credit both models with positional information neither had to learn.
FIRST_POSITION = 1


def collect_hidden(model: torch.nn.Module, windows: np.ndarray, device: str
                   ) -> list[np.ndarray]:
    """Capture the residual stream after the embedding and after every block.

    Args:
        model: A decoder LM in eval mode.
        windows: ``(N, T)`` token ids.
        device: Torch device.

    Returns:
        A list of ``n_layers + 1`` arrays of shape ``(N, T, C)``; element 0 is
        the embedding output, element ``i + 1`` the output of block ``i``.
    """
    taps: list[list[np.ndarray]] = [[] for _ in range(len(model.blocks) + 1)]
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # Blocks return (x, attention); the residual stream is the first item.
        tensor = output[0] if isinstance(output, tuple) else output
        captured.append(tensor.detach())

    handles = [block.register_forward_hook(hook) for block in model.blocks]
    embedding_out: list[torch.Tensor] = []

    def embed_hook(_module, _inputs, output):
        embedding_out.append(output.detach())

    handles.append(model.embedding_dropout.register_forward_hook(embed_hook))

    try:
        for start in range(0, len(windows), BATCH):
            chunk = torch.from_numpy(windows[start:start + BATCH]).to(device)
            captured.clear()
            embedding_out.clear()
            with torch.no_grad():
                model(chunk)
            taps[0].append(embedding_out[0].float().cpu().numpy())
            for i, tensor in enumerate(captured):
                taps[i + 1].append(tensor.float().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    return [np.concatenate(parts, axis=0) for parts in taps]


def ridge_decode(hidden: np.ndarray, n_train: int) -> tuple[float, float]:
    """Fit position from hidden states on one split, score on the other.

    Args:
        hidden: ``(N, T, C)`` residual stream.
        n_train: How many sequences form the training split.

    Returns:
        ``(R^2, mean absolute error in tokens)`` on the held-out sequences.
    """
    N, T, C = hidden.shape
    positions = np.arange(T, dtype=np.float64)[FIRST_POSITION:]

    def flatten(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = block[:, FIRST_POSITION:, :].reshape(-1, C).astype(np.float64)
        y = np.tile(positions, block.shape[0])
        return X, y

    X_tr, y_tr = flatten(hidden[:n_train])
    X_te, y_te = flatten(hidden[n_train:])

    # Centre on the training split only, then append a bias column.
    mean = X_tr.mean(0, keepdims=True)
    X_tr = np.hstack([X_tr - mean, np.ones((len(X_tr), 1))])
    X_te = np.hstack([X_te - mean, np.ones((len(X_te), 1))])

    penalty = RIDGE_LAMBDA * np.eye(X_tr.shape[1])
    penalty[-1, -1] = 0.0                      # never penalise the intercept
    weights = np.linalg.solve(X_tr.T @ X_tr + penalty, X_tr.T @ y_tr)

    prediction = X_te @ weights
    residual = ((y_te - prediction) ** 2).sum()
    total = ((y_te - y_te.mean()) ** 2).sum()
    return float(1.0 - residual / total), float(np.abs(y_te - prediction).mean())


def run(root: Path, device: str, report: Path | None) -> dict:
    """Decode position from every layer of both models.

    Args:
        root: Repository root.
        device: Torch device.
        report: Where to write JSON, or None.

    Returns:
        The results dictionary.
    """
    models = {
        "standard": ("16k", Paths.for_variant("hindi", root, "16k").final_model),
        "ablated": ("16k_nopos", Paths.for_variant("hindi", root, "16k_nopos").final_model),
    }

    paths = Paths.for_variant("hindi", root, "16k")
    stream = TokenStream(paths.split_bin("test"), SEQ_LEN)
    windows = sample_windows(stream, N_SEQUENCES, SEQ_LEN, PROBE_SEED)
    n_train = int(len(windows) * 0.75)
    LOGGER.info("stimulus: %d windows of %d tokens (%d train / %d test)",
                len(windows), SEQ_LEN, n_train, len(windows) - n_train)

    results: dict = {
        "seed": PROBE_SEED,
        "n_sequences": int(len(windows)),
        "seq_len": SEQ_LEN,
        "n_train_sequences": n_train,
        "ridge_lambda": RIDGE_LAMBDA,
        "first_position": FIRST_POSITION,
        "models": {},
    }

    for name, (variant, checkpoint) in models.items():
        model, config = load_trained("hindi", checkpoint, device, root, variant)
        hidden = collect_hidden(model, windows, device)
        taps = []
        for index, block in enumerate(hidden):
            r2, mae = ridge_decode(block, n_train)
            label = "embedding" if index == 0 else f"block{index - 1}"
            taps.append({"tap": label, "r2": r2, "mae_tokens": mae})
            LOGGER.info("%-9s %-10s R2 %6.3f  MAE %6.2f tokens",
                        name, label, r2, mae)
        results["models"][name] = {
            "variant": variant,
            "positional": config.positional,
            "taps": taps,
        }
        del model, hidden
        torch.cuda.empty_cache()

    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        LOGGER.info("report -> %s", report)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/ablation/position_decoding.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    run(args.root, args.device, args.report)


if __name__ == "__main__":
    main()
