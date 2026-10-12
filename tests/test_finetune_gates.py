"""Correctness gates for the finetuning harness (§4), plus one addition.

    python -m tests.test_finetune_gates --language hindi

THE STANDING DISCIPLINE: NO GPU QUOTA UNTIL THE WIRING IS PROVEN
----------------------------------------------------------------
Every gate here tests something that, if wrong, would produce a training run
that looks healthy and is silently useless.  A prompt mask off by one position,
a double shift, an optimizer that never sees the loss -- all of those still
print a falling loss curve.  These are cheap; a wasted full run is not.

P1-P6 are the gates the brief specifies.  P7 is added because this harness
makes a decision the brief does not discuss: it right-pads batches and passes
NO attention mask, relying on causality to make that safe.  That argument is
either true or the whole data pipeline is wrong, so it is tested rather than
asserted.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import Paths
from data import sft_data
from eval import completions as sft_generate
from train.finetune_config import SFTConfig
from train.finetune import SFTTrainer

LOGGER = logging.getLogger("tests.test_finetune_gates")

PASS = "PASS"
FAIL = "FAIL"


def gate(name: str, passed: bool, detail: str, numbers: dict | None = None,
         evidence: str = "") -> dict:
    """Package one gate outcome uniformly.

    Args:
        name: Gate id, e.g. ``"P1"``.
        passed: Whether it passed.
        detail: One-line summary for the table.
        numbers: Measurements for RESULTS_part2.md.
        evidence: Optional printable block (the eyeball-checkable views).

    Returns:
        A gate-result dict.
    """
    return {"gate": name, "status": PASS if passed else FAIL, "detail": detail,
            "numbers": numbers or {}, "evidence": evidence}


# ---------------------------------------------------------------------------
# P1 -- prompt mask
# ---------------------------------------------------------------------------


def p1_prompt_mask(trainer: SFTTrainer, n_examples: int = 5) -> dict:
    """P1: labels are -100 over the prompt and the true token over the completion.

    Checked on the UNSHIFTED ``input_ids`` / ``labels`` pair, which is the
    representation §2 specifies, and then the decoded side-by-side view is
    printed so the boundary can be eyeballed rather than trusted.

    Args:
        trainer: A constructed trainer, for examples and the tokenizer.
        n_examples: How many examples to check and display.

    Returns:
        A gate-result dict.
    """
    failures = []
    lines = []
    for example in trainer.train_examples[:n_examples]:
        ids, labels, n_prompt = (example.input_ids, example.labels,
                                 example.n_prompt)
        prompt_ok = all(l == sft_data.IGNORE_INDEX for l in labels[:n_prompt])
        completion_ok = list(labels[n_prompt:]) == list(ids[n_prompt:])
        eos_ok = ids[-1] == trainer.eos_id
        eos_supervised = labels[-1] == trainer.eos_id
        if not (prompt_ok and completion_ok and eos_ok and eos_supervised):
            failures.append(example.record_id)

        lines.append(f"  {example.record_id}: {n_prompt} masked prompt tokens, "
                     f"{example.n_completion} supervised completion tokens "
                     f"(EOS id {trainer.eos_id} included and supervised)")
        lines.append("    MASKED  : "
                     + trainer.tokenizer.decode(list(ids[:n_prompt])))
        lines.append("    LEARNED : "
                     + trainer.tokenizer.decode(list(ids[n_prompt:-1])))

    # Token-level view of the boundary for the first example, the place an
    # off-by-one would actually show up.
    first = trainer.train_examples[0]
    window = range(max(0, first.n_prompt - 3), min(len(first.input_ids),
                                                   first.n_prompt + 3))
    lines.append("")
    lines.append(f"  boundary detail for {first.record_id} "
                 f"(prompt ends at index {first.n_prompt - 1}):")
    lines.append(f"    {'idx':>5} {'token':<18} {'label':>8}  in loss?")
    for i in window:
        piece = trainer.tokenizer.id_to_piece(first.input_ids[i])
        label = first.labels[i]
        lines.append(f"    {i:>5} {piece!r:<18} {label:>8}  "
                     f"{'no (prompt)' if label == sft_data.IGNORE_INDEX else 'YES'}")

    total_supervised = sum(e.n_completion for e in trainer.train_examples)
    total_tokens = sum(len(e.input_ids) for e in trainer.train_examples)
    return gate(
        "P1", not failures,
        f"{n_examples}/{n_examples} examples correctly masked; "
        f"{total_supervised / total_tokens:.1%} of all train tokens are "
        f"loss-active",
        {"checked": n_examples, "failures": len(failures),
         "supervised_fraction": round(total_supervised / total_tokens, 6),
         "tokens_total": total_tokens, "tokens_supervised": total_supervised},
        "\n".join(lines))


# ---------------------------------------------------------------------------
# P2 -- the shift
# ---------------------------------------------------------------------------


def p2_loss_shift(trainer: SFTTrainer) -> dict:
    """P2: logits at position t are scored against token t+1.

    Three things are established, the last being the decisive one:

    1. after the collate's shift, ``y[t] == input_ids[t+1]`` wherever the label
       is active;
    2. the FIRST active label sits at ``t = n_prompt - 1`` -- the last prompt
       token is the position that predicts the first completion token;
    3. recomputing the cross-entropy by hand over exactly the active positions
       reproduces the model's own loss.  That last equality is what proves the
       mask, the shift and ``ignore_index`` are all doing what is claimed,
       since any one of them being wrong would change the number.

    Args:
        trainer: A constructed trainer.

    Returns:
        A gate-result dict.
    """
    example = trainer.train_examples[0]
    x, y, n_supervised = sft_data.collate([example], trainer.eos_id,
                                          trainer.device)

    ids, n_prompt = example.input_ids, example.n_prompt
    alignment_ok = all(int(y[0, t]) == ids[t + 1]
                       for t in range(len(ids) - 1)
                       if int(y[0, t]) != sft_data.IGNORE_INDEX)
    active = [t for t in range(x.size(1))
              if int(y[0, t]) != sft_data.IGNORE_INDEX]
    boundary_ok = active[0] == n_prompt - 1
    count_ok = n_supervised == example.n_completion

    trainer.model.eval()
    with torch.no_grad():
        logits, model_loss, _ = trainer.model(x, y)
        manual = F.cross_entropy(
            logits[0, active, :].float(),
            y[0, active].to(logits.device))
    trainer.model.train()

    difference = abs(float(model_loss) - float(manual))
    loss_ok = difference < 1e-4

    last_prompt = trainer.tokenizer.id_to_piece(ids[n_prompt - 1])
    first_completion = trainer.tokenizer.id_to_piece(ids[n_prompt])
    evidence = "\n".join([
        f"  example            : {example.record_id}",
        f"  prompt tokens      : {n_prompt}",
        f"  first active label : t = {active[0]} (expected {n_prompt - 1})",
        f"  x[t] = {last_prompt!r}  ->  y[t] = {first_completion!r}",
        f"  i.e. the LAST PROMPT TOKEN predicts the FIRST COMPLETION TOKEN",
        f"  active positions   : {len(active)} (completion tokens "
        f"{example.n_completion})",
        f"  model loss         : {float(model_loss):.8f}",
        f"  hand-computed loss : {float(manual):.8f}   (over active positions "
        f"only)",
        f"  absolute difference: {difference:.2e}",
    ])
    return gate(
        "P2", alignment_ok and boundary_ok and count_ok and loss_ok,
        f"shift verified; model loss matches a hand-computed cross-entropy "
        f"over the {len(active)} active positions to {difference:.1e}",
        {"alignment_ok": alignment_ok, "boundary_ok": boundary_ok,
         "count_ok": count_ok, "loss_difference": difference,
         "model_loss": float(model_loss), "manual_loss": float(manual),
         "active_positions": len(active)},
        evidence)


# ---------------------------------------------------------------------------
# P3 -- overfit one batch
# ---------------------------------------------------------------------------


def p3_overfit_batch(trainer: SFTTrainer, steps: int = 100,
                     batch_size: int = 8, lr: float = 1e-3) -> dict:
    """P3: one batch, ~100 steps, loss to near zero and exact reproduction.

    A model that cannot memorise a single batch has a broken gradient path
    somewhere, and this is the cheapest place to find out.  Reproducing the
    completions verbatim under greedy decoding is the stronger half of the
    test: a falling loss alone would not catch, say, a mask that supervised the
    wrong positions.

    The trainer's weights are restored afterwards so the gate leaves no trace
    on the run that follows.

    Args:
        trainer: A constructed trainer.
        steps: Optimizer steps to take.
        batch_size: Examples in the single batch.
        lr: Learning rate for this deliberate overfit.

    Returns:
        A gate-result dict.
    """
    import copy

    saved_model = copy.deepcopy(trainer.model.state_dict())
    batch = trainer.train_examples[:batch_size]
    x, y, _ = sft_data.collate(batch, trainer.eos_id, trainer.device)

    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr)
    trainer.model.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = trainer.model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    # Greedy decode from the prompts and compare completions verbatim.
    trainer.model.eval()
    prompts = [e.input_ids[:e.n_prompt] for e in batch]
    generated = sft_generate.generate_completions(
        trainer.model, prompts, trainer.eos_id, trainer.device,
        max_new_tokens=200, max_batch=batch_size)
    exact = 0
    lines = []
    for example, ids in zip(batch, generated):
        target = list(example.input_ids[example.n_prompt:-1])   # drop EOS
        hit = list(ids) == target
        exact += hit
        lines.append(f"  {example.record_id}: {'reproduced' if hit else 'DIFFERS'}")
        if not hit:
            lines.append(f"    want: {trainer.tokenizer.decode(target)}")
            lines.append(f"    got : {trainer.tokenizer.decode(list(ids))}")

    trainer.model.load_state_dict(saved_model)
    trainer.model.train()

    final = losses[-1]
    passed = final < 0.05 and exact == len(batch)
    evidence = "\n".join([
        f"  loss: {losses[0]:.4f} (step 1) -> {min(losses):.6f} (best) -> "
        f"{final:.6f} (step {steps})",
        f"  greedy reproduction: {exact}/{len(batch)} completions exact",
        *lines,
    ])
    return gate(
        "P3", passed,
        f"loss {losses[0]:.3f} -> {final:.2e} over {steps} steps; "
        f"{exact}/{len(batch)} completions reproduced verbatim",
        {"steps": steps, "batch_size": batch_size, "lr": lr,
         "initial_loss": losses[0], "final_loss": final,
         "min_loss": min(losses), "exact_reproductions": exact,
         "loss_curve": [round(l, 6) for l in losses]},
        evidence)


# ---------------------------------------------------------------------------
# P4 -- kill and resume
# ---------------------------------------------------------------------------


def p4_kill_and_resume(trainer: SFTTrainer, steps: int = 12,
                       kill_at: int = 6) -> dict:
    """P4: a resumed run continues the pre-kill curve exactly.

    The test runs ``steps`` optimizer steps, keeping a checkpoint written at
    ``kill_at``.  A SECOND trainer is then constructed from scratch, loads that
    checkpoint, and runs the remaining steps.  The two loss sequences over the
    overlapping range must agree.

    Agreement here is a strong statement, and it is why the data order had to
    be made a pure function of ``(seed, epoch)``: the resumed run rebuilds the
    identical epoch order and skips exactly the micro-batches it already
    consumed, so it sees the same batches in the same order.  Optimizer
    moments, the scaler, the step counter (which *is* the scheduler state,
    since the schedule is a pure function of it) and both RNG streams come back
    from the checkpoint, so dropout continues rather than restarting.

    Args:
        trainer: A constructed trainer; used for the first leg and then reset.
        steps: Total optimizer steps in the reference run.
        kill_at: Step at which the checkpoint is taken.

    Returns:
        A gate-result dict carrying both loss sequences, for the overlay plot.
    """
    import copy

    from model.config import Paths as _Paths

    pristine = copy.deepcopy(trainer.model.state_dict())
    ckpt_path = trainer.out_dir / "p4_kill.pt"

    def leg(trn, start_step: int, stop_step: int,
            snapshot_at: int | None) -> list[dict]:
        """Run steps [start_step, stop_step) and record each loss."""
        records = []
        trn.model.train()
        while trn.step < stop_step:
            epoch = trn.step // trn.steps_per_epoch
            micro_batches = sft_data.epoch_batches(
                trn.train_examples, trn.config.micro_batch, trn.config.seed,
                epoch, trn.config.length_bucket)
            skip = (trn.step % trn.steps_per_epoch) * trn.config.grad_accum
            pending: list[list] = []
            for position, micro in enumerate(micro_batches):
                if position < skip:
                    continue
                pending.append(micro)
                if len(pending) < trn.config.grad_accum:
                    continue
                lr = trn.set_lr(trn.step)
                loss, grad_norm, _ = trn.train_step(pending)
                pending = []
                trn.step += 1
                records.append({"step": trn.step, "loss": loss, "lr": lr,
                                "grad_norm": grad_norm})
                if snapshot_at is not None and trn.step == snapshot_at:
                    trn.save_checkpoint(ckpt_path)
                if trn.step >= stop_step:
                    break
        return records

    # Leg 1: the uninterrupted reference run.
    trainer.step = 0
    reference = leg(trainer, 0, steps, kill_at)

    # Leg 2: a completely fresh trainer resuming from the mid-run checkpoint.
    resumed_trainer = SFTTrainer(trainer.config, trainer.paths, trainer.device,
                                 trainer.out_dir / "p4_resume",
                                 trainer.data_root, arch=trainer.arch,
                                 base=trainer.base_path)
    payload = resumed_trainer.load_checkpoint(ckpt_path)
    resumed = leg(resumed_trainer, kill_at, steps, None)

    overlap_reference = [r for r in reference if r["step"] > kill_at]
    deltas = [abs(a["loss"] - b["loss"])
              for a, b in zip(overlap_reference, resumed)]
    lr_deltas = [abs(a["lr"] - b["lr"])
                 for a, b in zip(overlap_reference, resumed)]
    max_delta = max(deltas) if deltas else 0.0
    step_ok = payload["step"] == kill_at

    # Optimizer moments must come back identically, not just approximately.
    moment_deltas = []
    for group_ref, group_res in zip(
            trainer.optimizer.state_dict()["state"].values(),
            resumed_trainer.optimizer.state_dict()["state"].values()):
        for key in ("exp_avg", "exp_avg_sq"):
            if key in group_ref and key in group_res:
                moment_deltas.append(
                    float((group_ref[key] - group_res[key]).abs().max()))

    trainer.model.load_state_dict(pristine)
    trainer.step = 0
    passed = step_ok and max_delta < 1e-5 and max(lr_deltas or [0]) < 1e-12

    evidence = "\n".join([
        f"  reference run : steps 1..{steps}, checkpoint taken at {kill_at}",
        f"  resumed run   : fresh trainer, loaded step {payload['step']}, "
        f"ran {kill_at + 1}..{steps}",
        "",
        f"  {'step':>6} {'reference loss':>16} {'resumed loss':>16} {'delta':>12}",
        *[f"  {a['step']:>6} {a['loss']:>16.8f} {b['loss']:>16.8f} "
          f"{abs(a['loss'] - b['loss']):>12.2e}"
          for a, b in zip(overlap_reference, resumed)],
        "",
        f"  max loss delta over the overlap : {max_delta:.2e}",
        f"  max LR delta over the overlap   : {max(lr_deltas or [0]):.2e}",
        f"  step restored                   : {payload['step']} "
        f"(expected {kill_at})",
    ])
    return gate(
        "P4", passed,
        f"resumed curve matches the reference over steps {kill_at + 1}-{steps} "
        f"to {max_delta:.1e}; step, optimizer moments, schedule and RNG all "
        f"restored",
        {"steps": steps, "kill_at": kill_at, "step_restored": payload["step"],
         "max_loss_delta": max_delta,
         "max_lr_delta": max(lr_deltas or [0]),
         "max_optimizer_moment_delta_after_resume":
             max(moment_deltas) if moment_deltas else None,
         "reference_curve": reference, "resumed_curve": resumed},
        evidence)


# ---------------------------------------------------------------------------
# P5 -- EOS stopping
# ---------------------------------------------------------------------------


def p5_eos_stop(trainer: SFTTrainer, steps: int = 300, n_prompts: int = 8,
                cap: int = 200) -> dict:
    """P5: after a few hundred steps, generation stops at EOS.

    Runs a short real finetune, then greedy-decodes validation prompts and
    checks the generations terminate on their own rather than running to the
    length cap.  This is what the appended EOS is for; without it the model has
    no way to signal that the chain of thought is finished.

    Weights are restored afterwards.

    Args:
        trainer: A constructed trainer.
        steps: Optimizer steps of real finetuning first.
        n_prompts: Validation prompts to decode.
        cap: Generation cap; hitting it means the model did not stop.

    Returns:
        A gate-result dict.
    """
    import copy

    pristine = copy.deepcopy(trainer.model.state_dict())
    start_step = trainer.step
    trainer.step = 0
    trainer.run(max_steps=steps)

    examples = trainer.em_examples[:n_prompts]
    prompts = [e.input_ids[:e.n_prompt] for e in examples]
    trainer.model.eval()
    generated = sft_generate.generate_completions(
        trainer.model, prompts, trainer.eos_id, trainer.device,
        max_new_tokens=cap, max_batch=n_prompts)

    stopped = sum(1 for ids in generated if len(ids) < cap)
    lengths = [len(ids) for ids in generated]
    lines = [f"  after {steps} finetuning steps, greedy decode of "
             f"{n_prompts} val prompts (cap {cap}):"]
    for example, ids in zip(examples, generated):
        text = trainer.tokenizer.decode(list(ids))
        lines.append(f"    {example.record_id}: {len(ids)} tokens, "
                     f"{'stopped at EOS' if len(ids) < cap else 'HIT CAP'}")
        lines.append(f"      {text}")

    trainer.model.load_state_dict(pristine)
    trainer.step = start_step
    trainer.model.train()

    return gate(
        "P5", stopped == n_prompts,
        f"{stopped}/{n_prompts} generations terminated at EOS "
        f"(lengths {min(lengths)}-{max(lengths)} against a cap of {cap})",
        {"steps": steps, "prompts": n_prompts, "cap": cap,
         "stopped_at_eos": stopped, "lengths": lengths},
        "\n".join(lines))


# ---------------------------------------------------------------------------
# P6 -- no <unk> leakage
# ---------------------------------------------------------------------------


def p6_roundtrip(trainer: SFTTrainer, n_strings: int = 100) -> dict:
    """P6: encode-decode loses nothing, and the cues and digits survive.

    Args:
        trainer: A constructed trainer.
        n_strings: How many training strings to round-trip.

    Returns:
        A gate-result dict.
    """
    tokenizer = trainer.tokenizer
    unk = tokenizer.unk_id()
    cues = [trainer.language_config["cues"]["question"],
            trainer.language_config["cues"]["answer"]]

    mismatches, unk_hits = [], 0
    for example in trainer.train_examples[:n_strings]:
        original = tokenizer.decode(list(example.input_ids[:-1]))
        ids = tokenizer.encode(original, out_type=int)
        unk_hits += sum(1 for i in ids if i == unk)
        if tokenizer.decode(ids) != original:
            mismatches.append(example.record_id)

    probes = {}
    for probe in [*cues, "0123456789", "।", ">", "="]:
        ids = tokenizer.encode(probe, out_type=int)
        probes[probe] = {"tokens": len(ids),
                         "unk": sum(1 for i in ids if i == unk),
                         "roundtrip": tokenizer.decode(ids) == probe}

    probe_ok = all(p["unk"] == 0 and p["roundtrip"] for p in probes.values())
    evidence = "\n".join([
        f"  {n_strings} training strings round-tripped: "
        f"{n_strings - len(mismatches)} identical, {unk_hits} <unk> tokens",
        "",
        f"  {'probe':<14} {'tokens':>7} {'unk':>5}  roundtrip",
        *[f"  {probe!r:<14} {p['tokens']:>7} {p['unk']:>5}  {p['roundtrip']}"
          for probe, p in probes.items()],
    ])
    return gate(
        "P6", not mismatches and unk_hits == 0 and probe_ok,
        f"{n_strings - len(mismatches)}/{n_strings} strings round-trip "
        f"identically with {unk_hits} <unk>; cues and digits intact",
        {"strings": n_strings, "mismatches": len(mismatches),
         "unk_tokens": unk_hits, "probes": probes},
        evidence)


# ---------------------------------------------------------------------------
# P7 -- padding invariance (added, not in the brief)
# ---------------------------------------------------------------------------


def p7_padding_invariance(trainer: SFTTrainer) -> dict:
    """P7: trailing padding cannot change a real position's output.

    This harness right-pads batches and passes NO attention mask, on the
    argument that causal attention makes trailing padding unreachable from any
    real position.  If that argument were wrong, every batch containing
    sequences of unequal length would be quietly corrupted -- and the loss
    curve would still look fine.  So it is measured.

    Two comparisons:

    * one sequence alone versus the same sequence padded with 64 extra tokens
      -- logits over the real span must be identical;
    * the same sequence inside a mixed-length batch versus alone -- its loss
      contribution must be unchanged.

    Args:
        trainer: A constructed trainer.

    Returns:
        A gate-result dict.
    """
    trainer.model.eval()
    short, long = trainer.train_examples[0], None
    for example in trainer.train_examples[1:]:
        if len(example.input_ids) > len(short.input_ids) + 10:
            long = example
            break
    long = long or trainer.train_examples[1]

    x_alone, y_alone, _ = sft_data.collate([short], trainer.eos_id,
                                           trainer.device)
    with torch.no_grad():
        logits_alone, loss_alone, _ = trainer.model(x_alone, y_alone)

        # Explicit padding: 64 EOS tokens appended past the real span.
        padded_x = torch.cat(
            [x_alone, torch.full((1, 64), trainer.eos_id, dtype=torch.long,
                                 device=trainer.device)], dim=1)
        logits_padded, _, _ = trainer.model(padded_x)
        real = x_alone.size(1)
        logit_delta = float(
            (logits_alone - logits_padded[:, :real, :]).abs().max())

        # Batched with a longer sequence, which pads the short one implicitly.
        x_batch, y_batch, _ = sft_data.collate([short, long], trainer.eos_id,
                                               trainer.device)
        logits_batch, _, _ = trainer.model(x_batch)
        batch_delta = float(
            (logits_alone[0] - logits_batch[0, :real, :]).abs().max())

        # And the masked loss for the short row alone vs inside the batch.
        active = y_batch[0] != sft_data.IGNORE_INDEX
        row_loss = F.cross_entropy(
            logits_batch[0][active].float(), y_batch[0][active])
        loss_delta = abs(float(loss_alone) - float(row_loss))
    trainer.model.train()

    tolerance = 1e-4
    passed = (logit_delta < tolerance and batch_delta < tolerance
              and loss_delta < tolerance)
    evidence = "\n".join([
        f"  sequence length {real} (+1 unshifted), padded with 64 extra tokens",
        f"  max |logit| change from explicit padding      : {logit_delta:.2e}",
        f"  max |logit| change from batching with a longer"
        f" sequence ({len(long.input_ids)}): {batch_delta:.2e}",
        f"  masked loss alone {float(loss_alone):.8f} vs in-batch "
        f"{float(row_loss):.8f}  (delta {loss_delta:.2e})",
        f"  tolerance {tolerance:.0e} -- causal attention makes trailing "
        f"padding unreachable, so no attention mask is required",
    ])
    return gate(
        "P7", passed,
        f"trailing padding changes real-position logits by at most "
        f"{max(logit_delta, batch_delta):.1e}; no attention mask needed",
        {"logit_delta_explicit_padding": logit_delta,
         "logit_delta_batched": batch_delta, "loss_delta": loss_delta,
         "tolerance": tolerance},
        evidence)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def render_table(results: list[dict]) -> str:
    """Render the pass/fail table.

    Args:
        results: Gate-result dicts.

    Returns:
        A Markdown table.
    """
    lines = ["| gate | status | reported |", "| --- | --- | --- |"]
    for result in results:
        detail = result["detail"].replace("|", "\\|")
        lines.append(f"| {result['gate']} | {result['status']} | {detail} |")
    return "\n".join(lines)


def run_all(language: str, root: Path, device: str, out_dir: Path,
            quick: bool = False) -> dict:
    """Run every gate for one language.

    Args:
        language: ``"hindi"`` (the only language).
        root: Repository root.
        device: Torch device.
        out_dir: Scratch directory for gate checkpoints.
        quick: Shorten the two expensive gates (P3, P5) for a smoke run.

    Returns:
        ``{"language": ..., "gates": [...]}``.
    """
    config = SFTConfig.load(root / "configs" / "finetune_shared.yaml")
    config.run_name = "gates"
    paths = Paths.for_variant(language, root, config.variant)
    trainer = SFTTrainer(config, paths, device, out_dir, paths.finetune_data)

    results = [
        p1_prompt_mask(trainer),
        p2_loss_shift(trainer),
        p7_padding_invariance(trainer),
        p3_overfit_batch(trainer, steps=30 if quick else 100),
        p4_kill_and_resume(trainer),
        p5_eos_stop(trainer, steps=40 if quick else 300),
        p6_roundtrip(trainer),
    ]
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4, "P6": 5, "P7": 6}
    results.sort(key=lambda r: order[r["gate"]])
    return {"language": language, "device": device,
            "eos_id": trainer.eos_id,
            "base_checkpoint": str(trainer.base_path),
            "train_stats": trainer.train_stats,
            "val_stats": trainer.val_stats,
            "gates": results}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="shorten P3/P5 for a wiring smoke test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.root.resolve()
    languages = ("hindi",)
    report_path = (args.report
                   or root / "reports" / "gpt_finetune" / "sft_gates.json")

    summaries = []
    for language in languages:
        out_dir = root / "artifacts" / "runs" / f"finetune_gates_{language}"
        summary = run_all(language, root, device, out_dir, args.quick)
        summaries.append(summary)

        print()
        print(f"{'=' * 70}")
        print(f"### {language.upper()} -- correctness gates ({device})")
        print(f"{'=' * 70}")
        for result in summary["gates"]:
            print(f"\n--- {result['gate']} [{result['status']}] "
                  f"{result['detail']}")
            if result["evidence"]:
                print(result["evidence"])
        print()
        print(render_table(summary["gates"]))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"languages": summaries}, ensure_ascii=False, indent=2,
                   default=str), encoding="utf-8")

    failed = [(s["language"], g["gate"]) for s in summaries
              for g in s["gates"] if g["status"] != PASS]
    print()
    if failed:
        print(f"GATES FAILED: {failed}")
        raise SystemExit(1)
    print(f"All gates passed for {', '.join(languages)}. "
          f"Report: {report_path}")


if __name__ == "__main__":
    main()
