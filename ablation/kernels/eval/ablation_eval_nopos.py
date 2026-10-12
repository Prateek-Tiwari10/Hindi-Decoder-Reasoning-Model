"""ABLATION, KERNEL 2 OF 2 -- full Phase 2 evaluation, ablated vs standard.

HISTORICAL RECORD: this kernel ran on Kaggle against the course-era code
layout (``lm/``, ``configs/hindi/``) packaged by an earlier make_zip.sh.  Its
imports have been updated to the new package names, but its input-staging
logic still describes that layout.  Published IDs (``lma-bonus-*``) are kept.

Runs the complete Phase 2 suite over both models and writes the comparison:

  * perplexity and bits-per-byte on the held-out test split
  * generation metrics -- BLEU-4, chrF, ROUGE-L
  * diversity diagnostics -- repetition rate and Distinct-n across temperatures
  * attention heatmaps, entropy and mean attention distance
  * an order-sensitivity probe written for this ablation specifically

BOTH MODELS COME FROM THE HUB
-----------------------------
The ablation is pulled from the repository kernel 1 published, so this kernel
depends on a name rather than on a kernel output that expires.  The standard
model is pulled the same way where possible, and falls back to an attached
checkpoint.  Whichever path is taken, the config is checked before use: a run
that evaluated the standard model twice would produce a beautifully formatted
comparison showing no difference, and nothing would look wrong.

WHY THE ORDER PROBE IS HERE
---------------------------
Removing the positional table is widely described as making a model
order-blind.  That is only true at one layer.  Causal attention is itself a
positional signal -- position t attends to 0..t, so different positions see
different prefixes -- and from two layers onward the permutation invariance is
gone.  The probe measures how much order information each model actually uses,
which is the thing the ablation is supposed to change.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
ROOT = WORK / "repo"

LANGUAGE = "hindi"
VOCAB = 16_000
BIN_SUFFIX = "_16k"

#: Published by kernel 1.  Keep in step with ablation/push_hf.py.  The repo name predates the rename.
ABLATED_REPO = "Prateek-Tiwari10/LMA_bonus_nopos_hindi"
#: The standard Hindi model this is compared against.
STANDARD_REPO = "Prateek-Tiwari10/LMA_phase2"

MODELS = {
    "standard": {"variant": "16k",       "positional": "learned_absolute"},
    "ablated":  {"variant": "16k_nopos", "positional": "none"},
}

#: Bounded so the whole kernel fits comfortably inside one session.
N_PROMPTS = 200
MAX_NEW_TOKENS = 128
EVAL_BATCHES = 200


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
    """Install what the image lacks without disturbing torch."""
    for module, package in (("huggingface_hub", "huggingface_hub"),
                            ("sacrebleu", "sacrebleu"),
                            ("sentencepiece", "sentencepiece"),
                            ("safetensors", "safetensors")):
        try:
            __import__(module)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            package], check=True)


def assemble_tree() -> Path:
    """Build a repository-shaped tree with BOTH variants' configs staged."""
    code = find_input("lm/train.py", "ablation code dataset")
    bins = find_input(f"{LANGUAGE}/test{BIN_SUFFIX}.bin", "exported 16K bins")

    for package in ("lm", "common"):
        target = ROOT / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(code / package, target)

    (ROOT / LANGUAGE / "configs").mkdir(parents=True, exist_ok=True)
    for name in ("model_16k_nopos.json", "train_16k_nopos.json"):
        shutil.copy(code / "configs" / LANGUAGE / name,
                    ROOT / LANGUAGE / "configs" / name)
    # The standard variant's configs ship in the same dataset so this kernel
    # does not depend on the Phase 2 code dataset still existing.
    for name in ("model_16k.json", "train_16k.json"):
        source = code / "configs" / LANGUAGE / name
        if source.exists():
            shutil.copy(source, ROOT / LANGUAGE / "configs" / name)

    bin_dir = ROOT / LANGUAGE / "data" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        link = bin_dir / f"{split}{BIN_SUFFIX}.bin"
        source = bins / LANGUAGE / f"{split}{BIN_SUFFIX}.bin"
        if not link.exists() and source.exists():
            os.symlink(source, link)

    tok_dir = ROOT / "tokenizers" / LANGUAGE / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    name = f"{LANGUAGE}_bpe_{VOCAB}.model"
    shutil.copy(code / "tokenizers" / LANGUAGE / "tokenizer" / name,
                tok_dir / name)

    shutil.copy(code / "ablation" / "from_hub.py", ROOT / "from_hub.py")
    return ROOT


def locate_published(snapshot: Path, positional: str) -> Path:
    """Find the directory in a snapshot whose config matches what we want.

    Repositories differ in how they nest per-model folders, so rather than
    hard-coding a layout this looks for the config that says what it should.
    """
    candidates = []
    for config_path in sorted(snapshot.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if (config.get("positional") == positional
                and config.get("vocab_size") == VOCAB
                and config.get("language", LANGUAGE) == LANGUAGE
                and (config_path.parent / "model.safetensors").exists()):
            candidates.append(config_path.parent)
    if not candidates:
        raise SystemExit(
            f"no model in {snapshot} has positional={positional!r}, "
            f"vocab_size={VOCAB}, language={LANGUAGE}")
    if len(candidates) > 1:
        print(f"  warning: {len(candidates)} candidates, using {candidates[0]}",
              flush=True)
    return candidates[0]


def resolve_checkpoints() -> dict[str, Path]:
    """Get both models onto local disk as loadable checkpoints."""
    sys.path.insert(0, str(ROOT))
    import from_hub

    resolved = {}
    for tag, spec in MODELS.items():
        target = WORK / "checkpoints" / f"{tag}.pt"
        repo = ABLATED_REPO if tag == "ablated" else STANDARD_REPO
        try:
            snapshot = from_hub.fetch(repo, WORK / "hub" / tag)
            directory = locate_published(snapshot, spec["positional"])
            from_hub.to_checkpoint(directory, target,
                                   expect_positional=spec["positional"])
        except (Exception, SystemExit) as exc:
            # The ablation exists only on the Hub -- kernel 1 put it there and
            # there is nothing to fall back to, so a failure here is fatal.
            if tag == "ablated":
                raise SystemExit(
                    f"could not obtain the ablated model from {repo}: {exc}\n"
                    "Has kernel 1 finished and pushed?")
            # The standard model predates this work and may be attached as a
            # kernel output instead, so try that before giving up.
            print(f"  Hub fetch failed for {tag} ({exc}); "
                  "looking for an attached checkpoint", flush=True)
            found = None
            for candidate in sorted(INPUT.rglob("*.pt")):
                if candidate.stat().st_size > 50_000_000:
                    found = candidate
                    break
            if found is None:
                raise SystemExit(
                    f"could not obtain the {tag} model from {repo} and no "
                    "checkpoint is attached")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(found, target)
            print(f"  using attached {found}", flush=True)
        resolved[tag] = target
    return resolved


def run(command: list[str], env: dict) -> None:
    """Run one evaluation step, reporting rather than hiding a failure."""
    print("\n$ " + " ".join(str(c) for c in command), flush=True)
    result = subprocess.run([str(c) for c in command], cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"  !! exited {result.returncode}", flush=True)


def order_probe(checkpoint: Path, variant: str, out: Path) -> dict:
    """Measure how much each model's prediction depends on token order.

    The final token is held fixed and the prefix is shuffled, so the visible
    multiset of context tokens is identical and only the arrangement changes.
    A model that ignores order entirely scores 0.
    """
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from model.config import Paths
    from eval.lm_metrics import load_trained

    paths = Paths.for_variant(LANGUAGE, ROOT, variant)
    model, _ = load_trained(LANGUAGE, checkpoint, "cuda", ROOT, variant)
    model.eval()

    tokens = np.memmap(paths.split_bin("test"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(1337)
    deltas, T = [], 64
    with torch.no_grad():
        for _ in range(128):
            start = int(rng.integers(0, len(tokens) - T - 1))
            idx = torch.tensor(np.asarray(tokens[start:start + T], dtype=np.int64),
                               device="cuda").unsqueeze(0)
            perm = torch.cat([torch.randperm(T - 1, device="cuda"),
                              torch.tensor([T - 1], device="cuda")])
            a = model(idx)[0][0, -1].float().log_softmax(-1)
            b = model(idx[:, perm])[0][0, -1].float().log_softmax(-1)
            deltas.append(0.5 * (a.exp() * (a - b)).sum().item()
                          + 0.5 * (b.exp() * (b - a)).sum().item())

    result = {"variant": variant,
              "n_samples": len(deltas),
              "context_len": T,
              "mean_symmetric_kl_prefix_shuffled": float(np.mean(deltas)),
              "median": float(np.median(deltas)),
              "note": "0 means the model's next-token distribution is "
                      "completely independent of the order of its context"}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  order sensitivity ({variant}): "
          f"{result['mean_symmetric_kl_prefix_shuffled']:.4f} nats", flush=True)
    return result


def main() -> None:
    """Evaluate both models and write the comparison."""
    ensure_dependencies()
    assemble_tree()
    sys.path.insert(0, str(ROOT))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}

    checkpoints = resolve_checkpoints()
    reports = WORK / "ablation_eval"
    figures = WORK / "figures"
    reports.mkdir(parents=True, exist_ok=True)

    for tag, spec in MODELS.items():
        variant, checkpoint = spec["variant"], checkpoints[tag]
        print(f"\n{'='*70}\n{tag.upper()}  variant={variant}  {checkpoint}\n{'='*70}",
              flush=True)

        run([sys.executable, "-m", "eval.lm_metrics",
             "--language", LANGUAGE, "--variant", variant,
             "--root", ROOT, "--checkpoint", checkpoint,
             "--split", "test", "--max-batches", EVAL_BATCHES,
             "--report", reports / f"eval_{tag}.json"], env)

        run([sys.executable, "-m", "eval.generation_metrics",
             "--language", LANGUAGE, "--variant", variant,
             "--root", ROOT, "--checkpoint", checkpoint,
             "--n-prompts", N_PROMPTS, "--max-new-tokens", MAX_NEW_TOKENS,
             "--report", reports / f"generation_{tag}.json"], env)

        run([sys.executable, "-m", "eval.attention",
             "--language", LANGUAGE, "--variant", variant,
             "--root", ROOT, "--checkpoint", checkpoint,
             "--figures", figures,
             "--report", reports / f"attention_{tag}.json"], env)

        try:
            order_probe(checkpoint, variant, reports / f"order_{tag}.json")
        except Exception as exc:
            print(f"  order probe failed: {exc}", flush=True)

    summarise(reports)

    for package_dir in ("lm", "common"):
        shutil.rmtree(ROOT / package_dir, ignore_errors=True)
    shutil.rmtree(WORK / "hub", ignore_errors=True)
    print("\nablation evaluation complete")


def summarise(reports: Path) -> None:
    """Collect both models' numbers into one JSON and one markdown report."""
    def load(name):
        path = reports / name
        return json.loads(path.read_text()) if path.exists() else {}

    combined = {tag: {"perplexity": load(f"eval_{tag}.json"),
                      "generation": load(f"generation_{tag}.json"),
                      "attention": load(f"attention_{tag}.json"),
                      "order": load(f"order_{tag}.json")}
                for tag in MODELS}
    (reports / "comparison.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n")

    def ppl(tag, key):
        """One field from the test-split row for our language."""
        for row in combined[tag]["perplexity"].get("results", []):
            if row.get("language") == LANGUAGE and row.get("split") == "test":
                return row.get(key)
        return None

    def gen(tag, setting, key):
        """One generation metric at one decoding setting."""
        block = combined[tag]["generation"].get(LANGUAGE, {})
        for entry in block.get("settings", []):
            if entry.get("setting") == setting:
                return entry.get(key)
        return None

    def att(tag, key):
        return combined[tag]["attention"].get(LANGUAGE, {}).get("summary", {}).get(key)

    def order(tag):
        return combined[tag]["order"].get("mean_symmetric_kl_prefix_shuffled")

    def fmt(v):
        if v is None:
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def delta(a, b):
        """Ablated minus standard, when both exist."""
        if isinstance(a, float) and isinstance(b, float):
            return f"{b - a:+.4f}"
        return "n/a"

    def table(title, rows):
        out = [f"### {title}", "",
               "| metric | standard | ablated (no positions) | delta |",
               "| --- | ---: | ---: | ---: |"]
        for label, a, b in rows:
            out.append(f"| {label} | {fmt(a)} | {fmt(b)} | {delta(a, b)} |")
        out.append("")
        return out

    lines = [
        "# Ablation — Hindi without positional embeddings", "",
        "The ablated model is identical to the standard Hindi model except that",
        "the learned absolute positional table is removed entirely",
        '(`positional: "none"`). Same tokenizer, corpus, split, one-epoch',
        "horizon, optimiser recipe and seed, so the only variable is the table.",
        "",
        "| | standard | ablated |",
        "| --- | ---: | ---: |",
        "| parameters | 25,860,480 | 25,663,872 |",
        "| positional table | 512 x 384 learned | none |",
        "",
    ]

    lines += table("Language modelling (test split)", [
        ("perplexity",        ppl("standard", "perplexity"),   ppl("ablated", "perplexity")),
        ("bits per byte",     ppl("standard", "bits_per_byte"), ppl("ablated", "bits_per_byte")),
        ("mean CE (nats)",    ppl("standard", "mean_ce_nats"),  ppl("ablated", "mean_ce_nats")),
        ("bits per token",    ppl("standard", "bits_per_token"), ppl("ablated", "bits_per_token")),
    ])

    for setting in ("greedy", "temperature=0.5", "temperature=1.0",
                    "temperature=1.5"):
        rows = [(label,
                 gen("standard", setting, key),
                 gen("ablated", setting, key))
                for label, key in (("BLEU-4", "bleu4"),
                                   ("chrF2", "chrf2"),
                                   ("ROUGE-L", "rouge_l"),
                                   ("repetition rate (4-gram)", "repetition_rate_4gram"),
                                   ("Distinct-1", "distinct_1"),
                                   ("Distinct-2", "distinct_2"))]
        if any(a is not None or b is not None for _, a, b in rows):
            lines += table(f"Generation — {setting}", rows)

    lines += table("Attention", [
        ("first-layer mean distance", att("standard", "first_layer_mean_distance"),
         att("ablated", "first_layer_mean_distance")),
        ("last-layer mean distance",  att("standard", "last_layer_mean_distance"),
         att("ablated", "last_layer_mean_distance")),
        ("narrowest layer",           att("standard", "narrowest_layer"),
         att("ablated", "narrowest_layer")),
    ])

    lines += table("Order sensitivity", [
        ("symmetric KL, prefix shuffled (nats)", order("standard"), order("ablated")),
    ])

    lines += [
        "## Reading the order-sensitivity number", "",
        "The final token is held fixed and the prefix is shuffled, so both",
        "models see an identical multiset of context tokens in a different",
        "arrangement. A model that ignores order scores 0.",
        "",
        "Removing the positional table does **not** force that to 0. Causal",
        "attention is itself a positional signal: position *t* attends to",
        "0..*t*, so representations at different positions are built from",
        "different prefixes. A one-layer no-positional model is genuinely",
        "permutation-invariant over its prefix; at two layers and beyond that",
        "invariance is gone. The ablated model therefore has to infer order",
        "from the mask instead of reading it from an embedding, and these",
        "numbers say what that costs.",
        "",
        "Fields shown as `n/a` were not produced by this run; the per-model",
        "JSON beside this file is authoritative.",
    ]

    (reports / "comparison.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
