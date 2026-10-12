"""Factored reasoning evaluation: every model x every test slice, resumably.

    python -m eval.reasoning_eval --models pretrained,ft_60k
    python -m eval.reasoning_eval --models llama_ft_s1 --checkpoint PATH

Slices come from ``data/reasoning/manifest_scaling.json``: six in-domain slices
(relation / numeric / mixed at depths 2-3, relation at held-out depths 4, 5, 6)
plus four unseen domains x four depth cuts.  Every slice uses HELD-OUT entity
names.  Hard-coding the slice list here would let the evaluation silently
disagree with the data, so the manifest is the only source of slices.

RESUMABILITY IS THE POINT
-------------------------
~33,000 greedy generations per model do not fit one Kaggle session.  One raw file per ``(model, slice)`` is written the moment that pair
finishes, and a pair whose file already exists is skipped on the next run.  A
killed session is therefore a resume, not a restart, and RESULTS is assembled
by re-reading the raw files rather than from anything held in memory.

THE PRETRAINED BASES ARE EVALUATED TOO
--------------------------------------
They have never seen the output format, so they are expected to score ~0 EM.
That is the floor every finetuned number is a delta against; without it, "82%
EM" has no scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from model.config import ModelConfig, Paths
from model import build_model
from data.reasoning.generator import load_language_config
from eval import reasoning_metrics as eval_metrics
from eval.reasoning_parser import CompletionParser
from eval.completions import generate_completions
from train.finetune import load_tokenizer, resolve_eos

LOGGER = logging.getLogger("eval.reasoning_eval")

#: §6: generate up to this many new tokens, further capped per batch so that
#: prompt + generation can never exceed the model's 512-token context.  Part A
#: measured the longest whole example at 314 tokens, so this is slack, not a
#: binding limit -- but a model that fails to terminate must be cut off by the
#: context, not by a cap that hides non-termination.
MAX_NEW_TOKENS = 400
CONTEXT_LEN = 512

#: Where each model tag's weights live, relative to the repo root.  The two
#: GPT checkpoints are the fixed baseline; other models are evaluated with
#: ``--checkpoint PATH --tag NAME``.  Tags match the raw files already in
#: ``reports/gpt_finetune/eval_raw`` (``hi_<tag>_<slice>.jsonl``).
CHECKPOINTS = {
    "pretrained": "artifacts/checkpoints/gpt/final_16k.pt",
    "ft_60k": "artifacts/checkpoints/gpt_ft/ft_60k/best.pt",
}

#: Prefix the manifest's paths were written with; the files now live under
#: ``artifacts/finetune_data/``.
MANIFEST_PREFIX = "phase3/data_perf/"


def code_of(language: str) -> str:
    """Two-letter code used in data paths and raw-file names (always ``hi``).

    Args:
        language: ``"hindi"``.

    Returns:
        ``"hi"``.
    """
    if language != "hindi":
        raise ValueError(f"Hindi-only repository, got {language!r}")
    return "hi"


def data_path(root: Path, manifest_path: str) -> Path:
    """Resolve a manifest-recorded path to its location under artifacts.

    Args:
        root: Repository root.
        manifest_path: Path as written in ``manifest_scaling.json``.

    Returns:
        Absolute path of that file under ``artifacts/finetune_data/``.
    """
    rel = manifest_path
    if rel.startswith(MANIFEST_PREFIX):
        rel = rel[len(MANIFEST_PREFIX):]
    return Paths.for_variant("hindi", root).finetune_data / rel


def load_manifest(root: Path) -> dict:
    """Read Part A's machine-readable index.

    Args:
        root: Repository root.

    Returns:
        The parsed manifest.

    Raises:
        SystemExit: If it is absent -- Part A has not been run, and guessing
            the slice list is exactly what this module refuses to do.
    """
    path = Paths.for_variant("hindi", root).reasoning_dir / "manifest_scaling.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Part A builds it; this evaluation reads its "
            f"slice list from there and will not hard-code one.")
    return json.loads(path.read_text(encoding="utf-8"))


def slices_for(manifest: dict, code: str) -> dict[str, dict]:
    """Every test slice for one language, keyed by a flat tag.

    Args:
        manifest: The Part A manifest.
        code: ``"hi"``.

    Returns:
        Tag to its manifest entry.  The tag is the manifest key's last
        component (``rel_id_d23``, ``xdom_speed_ood_d4``), which is what the
        raw filenames and the report tables use.
    """
    entries = manifest["languages"][code]["test"]
    return {key.split("/")[-1]: {**entry, "manifest_key": key}
            for key, entry in entries.items()}


def load_model(path: Path, device: str, language: str):
    """Load one checkpoint, asserting it is the carried-forward recipe.

    The guard is the one §1 of the brief demands: a 24K/448/7L checkpoint
    loads without complaint and produces numbers that mean nothing, so the
    architecture is checked before any generation is spent on it.

    Args:
        path: Checkpoint file.
        device: Torch device.
        language: Expected language, so one language's weights can never be
            evaluated against the other's tokenizer.

    Returns:
        ``(model, payload)`` with the model in eval mode.

    Raises:
        SystemExit: On a missing file or an architecture mismatch.
    """
    if not Path(path).exists():
        raise SystemExit(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])

    # Shared dimensions of every architecture in the study; the parameter
    # count is architecture-specific and is checked by build_model instead.
    expected = {"language": language, "vocab_size": 16000, "d_model": 384,
                "n_layers": 11, "n_heads": 6, "context_len": 512}
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        raise SystemExit(
            f"{path} is not the carried-forward 16K/384/11L recipe.\n"
            f"  expected: {expected}\n  actual:   {actual}\n"
            f"If this reads 24000/448/7 you loaded the 24k baseline.")

    model = build_model(config, device, arch=payload.get("arch", "gpt"))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def raw_path(out_dir: Path, language: str, tag: str, slice_tag: str) -> Path:
    """Where one ``(model, slice)`` pair's rows live.

    Args:
        out_dir: Root of the raw-result tree.
        language: Language name.
        tag: Model tag.
        slice_tag: Slice tag.

    Returns:
        The file path, per §6's naming.
    """
    return out_dir / f"{code_of(language)}_{tag}_{slice_tag}.jsonl"


def evaluate_pair(model, records: list[dict], tokenizer, parser, labels: dict,
                  eos_id: int, device: str, batch_size: int,
                  limit: int | None = None) -> list[dict]:
    """Generate and score one ``(model, slice)`` pair.

    Args:
        model: Loaded model in eval mode.
        records: The slice's records.
        tokenizer: That language's frozen 16K SentencePiece model.
        parser: Completion parser for the language.
        labels: Affirmative/negative labels.
        eos_id: Stop token.
        device: Torch device.
        batch_size: Generation batch size.
        limit: Score only the first N records, for the smoke gate.

    Returns:
        One score row per record, each carrying its generation.
    """
    if limit:
        records = records[:limit]
    prompts = [tokenizer.encode(r["prompt"], out_type=int) for r in records]

    # Cap so prompt + generation stays inside the context for the LONGEST
    # prompt in the set; a per-batch cap would vary the budget between
    # batches and make non-termination look like a property of the slice.
    budget = min(MAX_NEW_TOKENS, CONTEXT_LEN - max(len(p) for p in prompts))
    if budget <= 0:
        raise SystemExit(f"prompts exceed the context: longest "
                         f"{max(len(p) for p in prompts)} of {CONTEXT_LEN}")

    generated = generate_completions(model, prompts, eos_id, device,
                                     max_new_tokens=budget,
                                     max_batch=batch_size)
    rows = []
    for record, ids in zip(records, generated):
        text = tokenizer.decode(list(ids))
        parsed = parser.parse(text, record["id"])
        row = eval_metrics.score_record(record, text, parsed, labels,
                                        len(ids), budget)
        row["generated"] = text
        row["prompt"] = record["prompt"]
        row["max_new_tokens"] = budget
        rows.append(row)
    return rows


def evaluate_model(tag: str, language: str, root: Path, device: str,
                   batch_size: int, out_dir: Path, manifest: dict,
                   only_slices: list[str] | None = None,
                   limit: int | None = None,
                   checkpoint: Path | None = None) -> dict:
    """Evaluate one model across every slice the manifest lists.

    A pair whose raw file already exists is skipped without loading its
    records, so a resumed session pays nothing for work already done.

    Args:
        tag: Model tag (``pretrained`` or ``ft_<size>``).
        language: Language name.
        root: Repository root.
        device: Torch device.
        batch_size: Generation batch size.
        out_dir: Raw-result root.
        manifest: The Part A manifest.
        only_slices: Restrict to these slice tags, for the smoke gate.
        limit: Records per slice, for the smoke gate.
        checkpoint: Explicit checkpoint for a tag not in :data:`CHECKPOINTS`.

    Returns:
        A per-slice summary for this model.
    """
    code = code_of(language)
    wanted = slices_for(manifest, code)
    if only_slices:
        wanted = {t: e for t, e in wanted.items() if t in only_slices}

    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [t for t in wanted
               if not raw_path(out_dir, language, tag, t).exists()]
    if not pending:
        LOGGER.info("%s/%s: all %d slices already evaluated, skipping load",
                    language, tag, len(wanted))
    else:
        model, payload = load_model(
            checkpoint or root / CHECKPOINTS[tag], device, language)
        paths = Paths.for_variant(language, root, "16k")
        tokenizer = load_tokenizer(paths)
        eos_id = resolve_eos(tokenizer)
        config = load_language_config(paths.reasoning_config)
        parser = CompletionParser(config)
        labels = config["labels"]

        for slice_tag in pending:
            entry = wanted[slice_tag]
            records = [json.loads(line) for line in
                       data_path(root, entry["path"]).read_text(
                           encoding="utf-8").splitlines() if line]
            started = time.time()
            rows = evaluate_pair(model, records, tokenizer, parser, labels,
                                 eos_id, device, batch_size, limit)
            for row in rows:
                row["model"] = tag
                row["language"] = language
                row["unseen_domain"] = entry["unseen_domain"]
            destination = raw_path(out_dir, language, tag, slice_tag)
            with open(destination, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = eval_metrics.summarize(rows)
            LOGGER.info(
                "%s/%-12s %-28s EM %.4f chain %.4f fmt %.4f copy %s (%.0fs)",
                language, tag, slice_tag, summary["em"],
                summary["chain_correct"], summary["format_valid"],
                "n/a" if summary["copy_rate"] is None
                else f"{summary['copy_rate']:.4f}",
                time.time() - started)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {tag: {t: eval_metrics.summarize(
        [json.loads(line) for line in
         raw_path(out_dir, language, tag, t).read_text(
             encoding="utf-8").splitlines() if line])
        for t in wanted}}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--models", default="all",
                        help="'all', or comma-separated tags "
                             "(pretrained,ft_10k,...)")
    parser.add_argument("--slices", default=None,
                        help="comma-separated slice tags; default every slice "
                             "in the manifest")
    parser.add_argument("--limit", type=int, default=None,
                        help="records per slice — the smoke gate uses 100")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None,
                        help="raw-result root (default reports/gpt_finetune/eval_raw)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="evaluate this checkpoint under --models TAG")
    args = parser.parse_args()

    root = args.root.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or root / "reports" / "gpt_finetune" / "eval_raw"
    manifest = load_manifest(root)

    languages = ("hindi",)
    tags = (list(CHECKPOINTS) if args.models == "all"
            else [t.strip() for t in args.models.split(",")])
    unknown = [t for t in tags if t not in CHECKPOINTS]
    if args.checkpoint is not None:
        if len(tags) != 1:
            raise SystemExit("--checkpoint needs exactly one --models tag")
        unknown = []
    if unknown:
        raise SystemExit(f"unknown model tag(s): {unknown}; "
                         f"known: {list(CHECKPOINTS)}")
    only = [s.strip() for s in args.slices.split(",")] if args.slices else None

    summaries: dict = {}
    started = time.time()
    for language in languages:
        summaries[language] = {}
        for tag in tags:
            summaries[language].update(
                evaluate_model(tag, language, root, device, args.batch_size,
                               out_dir, manifest, only, args.limit,
                               args.checkpoint))

    summary_path = out_dir / "summary_matrix.json"
    summary_path.write_text(
        json.dumps({"models": tags, "languages": list(languages),
                    "limit": args.limit, "summaries": summaries},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nraw rows:  {out_dir}")
    print(f"summary:   {summary_path}")
    print(f"wall clock: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
