"""ABLATION, KERNEL 1 OF 2 -- train Hindi with no positional embeddings.

HISTORICAL RECORD: this kernel ran on Kaggle against the course-era code
layout (``lm/``, ``configs/hindi/``) packaged by an earlier make_zip.sh.  Its
imports have been updated to the new package names, but its input-staging
logic still describes that layout.  Published IDs (``lma-bonus-*``) are kept.

WHAT THIS RUN IS FOR
--------------------
The ablation retrains one language retrained with positional embeddings removed,
then evaluated with the full Phase 2 suite against the standard model.  This
kernel trains it and publishes it; kernel 2 evaluates it.  Splitting them means
a bug in evaluation cannot cost five hours of training, and it is why the model
goes to the Hub at the end of this script rather than being handed over as a
kernel output.

THE COMPARISON IS CONTROLLED
----------------------------
Same tokenizer, same corpus, same split, same one-epoch horizon, same optimiser
recipe (peak LR 3e-3, 750 warmup, 32 x 2 x 512 per step), same seed.  The only
difference is ``positional: "none"`` in the model config.  Anything else that
changed would make the loss difference unattributable.

BUDGET
------
The standard 16K/384/11L model measured 28,992 tokens/sec on a T4.  One epoch
is 528,384,000 tokens, so expect roughly 5.1 hours.  Removing the positional
table drops 196,608 parameters (0.76%), which does not meaningfully change
throughput.  Checkpoints land every 500 steps, so a timeout costs minutes, not
the run -- re-push with this kernel's own output attached to resume.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
ROOT = WORK / "repo"

#: The ablation variant.  See ``model.config.VARIANTS``.
VARIANT = "16k_nopos"
#: The token bins are the SAME ones the standard 16K model trained on.
BIN_SUFFIX = "_16k"
VOCAB = 16_000
LANGUAGE = "hindi"

#: Identical to the standard run.  Not re-swept: re-tuning the ablation would
#: make the loss gap a mixture of "no positions" and "different recipe".
PEAK_LR = 0.003
WARMUP_STEPS = 750
MICRO_BATCH = 32
GRAD_ACCUM = 2
CONTEXT_LEN = 512
TOKENS_PER_STEP = MICRO_BATCH * GRAD_ACCUM * CONTEXT_LEN     # 32,768


def find_input(relative: str, what: str) -> Path:
    """Locate the mounted input directory containing ``relative``.

    Searched recursively, at any depth.  A dataset built from an archive keeps
    whatever top-level directory the archive had, and a notebook output mounts
    at ``/kaggle/input/notebooks/<owner>/<slug>/output/`` -- five levels down.
    A fixed-depth search works for one layout and silently fails on the other,
    so the marker file is what is searched for, not the path to it.
    """
    for candidate in sorted(INPUT.rglob(relative)):
        root = Path(str(candidate)[:-len(relative)].rstrip("/"))
        print(f"found {what}: {root}", flush=True)
        return root
    listing = [str(p) for p in sorted(INPUT.rglob("*")) if p.is_dir()][:80]
    raise SystemExit(f"Could not find {what} (marker {relative!r}).\n"
                     "Mounted directories:\n  " + "\n  ".join(listing))


def ensure_dependencies() -> None:
    """Install what the image lacks, quietly and without upgrading torch."""
    for module, package in (("huggingface_hub", "huggingface_hub"),
                            ("sentencepiece", "sentencepiece")):
        try:
            __import__(module)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--no-deps", package], check=True)


def verify_gpu() -> None:
    """Fail now, not in five hours, if the GPU cannot run this build."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4.")
    name = torch.cuda.get_device_name(0)
    try:
        torch.zeros(8, device="cuda").add_(1).sum().item()
    except RuntimeError as exc:
        raise SystemExit(f"{name} cannot run this PyTorch build: {exc}")
    print(f"GPU verified: {name}", flush=True)


def assemble_tree() -> Path:
    """Build a repository-shaped tree from the attached inputs.

    Only the ablation's configs are staged.  If the standard 16K configs were
    also present, a typo in VARIANT would silently fall back to them and train
    the wrong architecture -- which would look like a successful run.
    """
    code = find_input("lm/train.py", "ablation code dataset")
    bins = find_input(f"{LANGUAGE}/train{BIN_SUFFIX}.bin",
                      "exported 16K token bins")

    for package in ("lm", "common"):
        target = ROOT / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(code / package, target)

    (ROOT / LANGUAGE / "configs").mkdir(parents=True, exist_ok=True)
    for name in (f"model_{VARIANT}.json", f"train_{VARIANT}.json"):
        source = code / "configs" / LANGUAGE / name
        if not source.exists():
            raise SystemExit(f"{source} missing from the code dataset")
        shutil.copy(source, ROOT / LANGUAGE / "configs" / name)

    # Symlinked, not copied: ~1 GB, and the read-only mount memmaps fine.
    bin_dir = ROOT / LANGUAGE / "data" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        link = bin_dir / f"{split}{BIN_SUFFIX}.bin"
        if not link.exists():
            os.symlink(bins / LANGUAGE / f"{split}{BIN_SUFFIX}.bin", link)

    tok_dir = ROOT / "tokenizers" / LANGUAGE / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    name = f"{LANGUAGE}_bpe_{VOCAB}.model"
    shutil.copy(code / "tokenizers" / LANGUAGE / "tokenizer" / name,
                tok_dir / name)

    shutil.copy(code / "ablation" / "push_hf.py", ROOT / "push_hf.py")
    return ROOT


def restore_previous_output() -> None:
    """Copy a prior run's checkpoints back in so a re-push resumes.

    ``/kaggle/working`` starts empty on every session.  Without this, a kernel
    re-pushed after a timeout finds no ``last.pt`` and restarts from scratch
    while appearing to work perfectly.
    """
    out = WORK / f"train_{VARIANT}_{LANGUAGE}"
    for base in sorted(INPUT.glob("*")):
        previous = base / f"train_{VARIANT}_{LANGUAGE}"
        if previous.is_dir() and (previous / "last.pt").exists():
            out.mkdir(parents=True, exist_ok=True)
            for item in previous.iterdir():
                target = out / item.name
                if not target.exists():
                    shutil.copy(item, target)
            print(f"resuming from {previous} "
                  f"({', '.join(sorted(p.name for p in out.iterdir()))})",
                  flush=True)
            return
    print("no previous output attached; starting from scratch", flush=True)


def write_config() -> int:
    """Pin the recipe into the config the trainer reads. Returns the step count."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from model.config import ModelConfig, Paths, TrainConfig

    paths = Paths.for_variant(LANGUAGE, ROOT, VARIANT)
    model = ModelConfig.load(paths.model_config)
    if model.positional != "none":
        raise SystemExit(
            f"model config says positional={model.positional!r}; this kernel "
            "exists to train the ablation and refuses to train the baseline")
    print(f"variant {VARIANT}: vocab={model.vocab_size} d_model={model.d_model} "
          f"n_layers={model.n_layers} n_heads={model.n_heads} "
          f"positional={model.positional!r} params={model.parameter_count:,} "
          f"in_budget={model.within_budget}", flush=True)

    train_tokens = len(np.memmap(paths.split_bin("train"),
                                 dtype=np.uint16, mode="r"))
    steps = train_tokens // TOKENS_PER_STEP
    print(f"{LANGUAGE}: {train_tokens:,} tokens / {TOKENS_PER_STEP:,} per step "
          f"= {steps:,} steps (one epoch)", flush=True)
    print(f"estimated training time at 28,992 tok/s: "
          f"{steps * TOKENS_PER_STEP / 28992 / 3600:.2f} h", flush=True)

    config = TrainConfig.from_dict({
        **TrainConfig.load(paths.train_config).to_dict(),
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": steps,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "eval_interval": 250,
        "ckpt_interval": 500,
        "run_name": f"ablation_{VARIANT}_{LANGUAGE}",
    })
    config.save(paths.train_config)
    return steps


def publish(out: Path, env: dict) -> None:
    """Package the trained weights and push them to the Hub."""
    import json
    sys.path.insert(0, str(ROOT))
    from train.hub import package, verify
    import push_hf

    summary_path = WORK / f"ablation_train_{VARIANT}_{LANGUAGE}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    model_dir = WORK / "hf_ablation_nopos"
    package(LANGUAGE, VARIANT, model_dir, root=ROOT, checkpoint=out / "best.pt")

    checks = verify(model_dir, root=ROOT)
    print(f"package verified: {checks}", flush=True)

    url = push_hf.push(model_dir, summary)
    (WORK / "hf_push_ablation.json").write_text(
        json.dumps({"repo_url": url, "variant": VARIANT,
                    "language": LANGUAGE, "summary": summary}, indent=2) + "\n")


def main() -> None:
    """Train one epoch without positional embeddings, then publish."""
    ensure_dependencies()
    assemble_tree()
    sys.path.insert(0, str(ROOT))
    verify_gpu()
    restore_previous_output()
    steps = write_config()

    out = WORK / f"train_{VARIANT}_{LANGUAGE}"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}

    command = [sys.executable, "-m", "train.pretrain",
               "--language", LANGUAGE,
               "--variant", VARIANT,
               "--root", str(ROOT),
               "--out", str(out)]
    print("\n" + " ".join(command) + "\n", flush=True)
    result = subprocess.run(command, cwd=str(ROOT), env=env)

    # Rendered even on failure: a partial curve is what decides whether to
    # resume or to change something.
    log = out / "log.jsonl"
    if log.exists():
        horizon = (f"Ablation (no positional embeddings) -- one epoch, "
                   f"{steps:,} steps x {TOKENS_PER_STEP:,} tokens")
        subprocess.run([sys.executable, "-m", "viz.pretrain_curves",
                        "--language", LANGUAGE,
                        "--log", str(log),
                        "--out", str(WORK / "figures"),
                        "--peak-lr", str(PEAK_LR),
                        "--tag", f"_{VARIANT}",
                        "--horizon", horizon,
                        "--report", str(WORK / f"ablation_train_{VARIANT}_{LANGUAGE}.json")],
                       cwd=str(ROOT), env=env)

    if result.returncode != 0:
        raise SystemExit(f"training failed with exit code {result.returncode}")

    publish(out, env)

    # The assembled source tree would otherwise be saved as kernel output.
    for package_dir in ("lm", "common"):
        shutil.rmtree(ROOT / package_dir, ignore_errors=True)
    shutil.rmtree(WORK / "hf_ablation_nopos", ignore_errors=True)

    print(f"\nablation training complete: {steps:,} steps, model on the Hub")


if __name__ == "__main__":
    main()
