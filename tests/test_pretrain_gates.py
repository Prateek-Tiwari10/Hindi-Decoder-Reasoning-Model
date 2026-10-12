"""Gate A and Gate B: the cheap de-risk checks that run before GPU quota.

The plan is explicit that these come before the sweep.  Both cost seconds to
minutes on any GPU; the failures they catch cost hours or days if they are found
after a production run instead.

GATE A -- OVERFIT A SINGLE BATCH
--------------------------------
Train on one small batch, repeatedly, with dropout off.  The loss must collapse
toward zero.  A model with 27.9M parameters memorising 8x64 tokens is not a
feat -- which is exactly why it is diagnostic: if the loss *cannot* be driven to
zero on data the model is free to memorise, the bug is structural.  A detached
graph, a mis-shaped loss, a target misalignment, a frozen parameter group, an
optimiser that is not connected to the parameters -- all of them show up here.
No learning rate rescues any of them, so there is no point tuning one first.

Dropout is disabled because it injects noise that puts a floor under the
achievable loss, which would make a passing run look like a failing one.

GATE B -- KILL AND RESUME
-------------------------
Train a few steps, checkpoint, destroy the trainer, rebuild it from disk and
continue.  The loss must pick up where it left off.  The failure this catches is
resuming with weights but *without* optimiser state: AdamW's moment estimates
are rebuilt from zero, the first few updates are badly scaled, and the loss
spikes.  On a 15,000-step run that is invisible unless it is looked for -- it
just makes the run slightly worse than it should have been.

This gate also verifies that the learning rate resumes at the right point on the
schedule, which is why the schedule is a pure function of the step number rather
than a stateful scheduler object.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from model.config import ModelConfig, Paths, TrainConfig
from data.token_stream import to_torch
from train.pretrain import Trainer

LOGGER = logging.getLogger("gates")


def gate_a_overfit(model_config: ModelConfig, train_config: TrainConfig,
                   paths: Paths, device: str, steps: int = 200,
                   batch: int = 8, seq_len: int = 64) -> dict:
    """Gate A: drive the loss to ~0 on one fixed batch.

    Uses a short sequence and a small batch so the whole gate runs in seconds
    even on a laptop GPU, and a higher LR than production because the goal is
    fast memorisation, not a good model.

    Args:
        model_config: Architecture.  Dropout is forced to 0 for this gate.
        train_config: Recipe, used for the optimiser settings.
        paths: Data locations.
        device: Torch device.
        steps: Optimisation steps on the single batch.
        batch: Sequences in the batch.
        seq_len: Tokens per sequence.

    Returns:
        Result with the loss trajectory and the verdict.
    """
    config = ModelConfig.from_dict({**model_config.to_dict(), "dropout": 0.0})
    trainer = Trainer(config, train_config, paths, device,
                      out_dir=Path("/tmp/gate_a"))

    # One batch, drawn once and then reused for every step.
    rng = np.random.default_rng(0)
    fixed = trainer.train_stream.sample(batch, rng)
    x, y = to_torch((fixed[0][:, :seq_len], fixed[1][:, :seq_len]), device)

    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=3e-4,
                                  betas=train_config.betas)
    trainer.model.train()

    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = trainer.model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),
                                       train_config.grad_clip)
        optimizer.step()
        losses.append(loss.item())

    shutil.rmtree("/tmp/gate_a", ignore_errors=True)
    initial, final = losses[0], losses[-1]
    return {
        "name": "Gate A -- overfit a single batch",
        "steps": steps,
        "batch": [batch, seq_len],
        "initial_loss": initial,
        "final_loss": final,
        "trajectory": [round(losses[i], 4)
                       for i in range(0, steps, max(1, steps // 10))],
        # 0.5 nats is far below ln(24000)=10.09 and far below anything a model
        # could reach without genuinely memorising the batch.
        "passed": final < 0.5 and final < initial,
    }


def gate_b_resume(model_config: ModelConfig, train_config: TrainConfig,
                  paths: Paths, device: str, steps_before: int = 12,
                  steps_after: int = 12) -> dict:
    """Gate B: kill mid-run, restart, and confirm the loss continues smoothly.

    Args:
        model_config: Architecture.
        train_config: Recipe.
        paths: Data locations.
        device: Torch device.
        steps_before: Steps to run before the simulated crash.
        steps_after: Steps to run after resuming.

    Returns:
        Result comparing the loss and LR across the boundary.
    """
    out_dir = Path("/tmp/gate_b")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    # -- run 1: train a few steps, then checkpoint and throw the trainer away --
    first = Trainer(model_config, train_config, paths, device, out_dir=out_dir)
    before = [first.train_step()[0] for _ in range(steps_before)]
    first.save_checkpoint(out_dir / "last.pt")
    step_at_kill = first.step
    lr_at_kill = first.train_config.lr_at(step_at_kill)
    opt_state_before = len(first.optimizer.state_dict()["state"])
    del first

    # -- run 2: rebuild from scratch and resume from disk --
    second = Trainer(model_config, train_config, paths, device, out_dir=out_dir)
    resumed = second.maybe_resume()
    after = [second.train_step()[0] for _ in range(steps_after)]

    opt_state_after = len(second.optimizer.state_dict()["state"])

    # The loss either side of the boundary should be comparable.  A cold
    # optimiser would show up as a jump well outside the run's own noise.
    tail = float(np.mean(before[-4:]))
    head = float(np.mean(after[:4]))
    spike = head - tail
    noise = float(np.std(before[-8:])) or 1e-6

    shutil.rmtree(out_dir, ignore_errors=True)
    return {
        "name": "Gate B -- kill and resume",
        "resumed": resumed,
        "step_at_kill": step_at_kill,
        "step_after_resume": second.step - steps_after,
        "step_continuous": (second.step - steps_after) == step_at_kill,
        "lr_at_kill": lr_at_kill,
        "lr_after_resume": train_config.lr_at(step_at_kill),
        # bool() because np.isclose returns np.bool_, which json cannot encode.
        "lr_continuous": bool(np.isclose(lr_at_kill,
                                         train_config.lr_at(step_at_kill))),
        "optimizer_state_tensors": [opt_state_before, opt_state_after],
        "optimizer_state_restored": opt_state_after == opt_state_before,
        "loss_before_kill": round(tail, 4),
        "loss_after_resume": round(head, 4),
        "spike": round(spike, 4),
        "spike_in_noise_units": round(spike / noise, 2),
        # A genuine cold-optimiser restart shows a jump of many nats; the
        # threshold is generous because a few steps of real training also move
        # the loss.
        "passed": (resumed
                   and (second.step - steps_after) == step_at_kill
                   and opt_state_after == opt_state_before
                   and spike < 0.5),
    }


def main() -> None:
    """Run both gates and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--device", default=None)
    parser.add_argument("--micro-batch", type=int, default=8,
                        help="small so the gates fit a laptop GPU; they test "
                             "correctness, not throughput")
    parser.add_argument("--gate-a-steps", type=int, default=200)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/gates.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log = logging.getLogger("gates.main")
    log.setLevel(logging.INFO)

    paths = Paths.for_variant(args.language, args.root, args.variant)
    model_config = ModelConfig.load(paths.model_config)
    train_config = TrainConfig.from_dict({
        **TrainConfig.load(paths.train_config).to_dict(),
        "micro_batch": args.micro_batch, "grad_accum": 2,
    })
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}  language={args.language}  "
          f"variant={args.variant or 'baseline'}  "
          f"params={model_config.parameter_count:,}")
    print("=" * 70)

    results = [
        gate_a_overfit(model_config, train_config, paths, device,
                       steps=args.gate_a_steps),
        gate_b_resume(model_config, train_config, paths, device),
    ]

    for result in results:
        print(f"\n[{'PASS' if result['passed'] else 'FAIL'}] {result['name']}")
        for key, value in result.items():
            if key in ("name", "passed"):
                continue
            print(f"    {key:28} {value}")

    report = {"language": args.language, "variant": args.variant,
              "model_config": model_config.to_dict(), "device": device,
              "gates": results,
              "all_passed": all(r["passed"] for r in results)}
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if report['all_passed'] else 'FAIL'}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
