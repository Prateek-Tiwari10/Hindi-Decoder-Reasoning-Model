"""CLI: build the Phase 3 reasoning datasets and run every self-check.

    python -m data.reasoning.run --language both

Generation, verification and reporting are kept apart on purpose.
``generator.py`` builds records and knows nothing about checking them;
``checks.py`` verifies records and never generates one; ``report.py`` renders
and computes nothing.  This module is the only place the three meet, and it
writes two artifacts: the dataset files under ``artifacts/finetune_data/part1/<code>/`` and a
machine-readable summary that ``report.py`` turns into RESULTS_part1.md.

NO MODEL IS LOADED ANYWHERE IN PART 1.  The Phase 2 tokenizers are opened
read-only, and only to measure sequence lengths -- never retrained, never
modified, never used to write token ids (§12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:                     # allow `python run.py` too
    sys.path.insert(0, str(_HERE))

import checks                                       # noqa: E402
import generator                                    # noqa: E402

#: The Phase 2 context length.  Every example must fit well inside it (§0, G5).
CONTEXT_LEN = 512

#: Sequence length above which a slice is flagged even though it still fits.
WARN_AT = 400


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize(records: Sequence[dict]) -> str:
    """Render records as JSONL.

    ``ensure_ascii=False`` is required, not cosmetic: escaping Devanagari to
    ``\\uXXXX`` would quadruple the file size and make the data unreadable in
    review, which is the one thing a human has to do with it before training.

    Args:
        records: Records for one slice.

    Returns:
        The JSONL text, newline-terminated.
    """
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def digest(text: str) -> str:
    """SHA-256 of a UTF-8 string, for the determinism check.

    Args:
        text: Serialized slice content.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    """Path relative to the repo root, falling back to absolute.

    Args:
        path: The file written.
        root: Repository root.

    Returns:
        A display path.  Output outside the repo (a scratch run, say) is
        reported absolute rather than raising.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def output_name(slice_tag: str, spec: dict) -> str:
    """File name for one slice, following the §2 layout.

    Args:
        slice_tag: The slice's name.
        spec: Its definition, for the split it belongs to.

    Returns:
        e.g. ``"train.jsonl"``, ``"val.jsonl"``, ``"test_rel_ood_d4.jsonl"``.
    """
    if "filename" in spec:
        return spec["filename"]
    split = spec["split"]
    if split in ("train", "val"):
        return f"{split}.jsonl"
    return f"test_{slice_tag}.jsonl"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_language(config: dict, build_config: dict
                   ) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Generate every slice for one language.

    The ``seen_prompts`` registry is threaded through every slice in one pass,
    which is what enforces "no prompt in more than one split" (§8) rather than
    checking for it afterwards and having to discard whole records.

    Args:
        config: Language config.
        build_config: The build plan.

    Returns:
        ``(all_records, records_by_slice, slice_stats)``.
    """
    seen_prompts: dict[str, str] = {}
    by_slice: dict[str, list[dict]] = {}
    stats: dict[str, dict] = {}
    for slice_tag in build_config["slices"]:
        records, slice_stats = generator.generate_slice(
            config, build_config, slice_tag, seen_prompts)
        by_slice[slice_tag] = records
        stats[slice_tag] = slice_stats
    all_records = [r for tag in build_config["slices"] for r in by_slice[tag]]
    return all_records, by_slice, stats


# ---------------------------------------------------------------------------
# Analysis -- computation only.  Rendering lives in report.py.
# ---------------------------------------------------------------------------


def count_table(by_slice: dict[str, list[dict]]) -> dict:
    """Per-slice counts broken down by family, depth and question type (§11.1).

    Args:
        by_slice: Records grouped by slice tag.

    Returns:
        Slice tag to a dict of split, size and three breakdown counters.
    """
    table = {}
    for tag, records in by_slice.items():
        table[tag] = {
            "split": records[0]["split"] if records else "",
            "size": len(records),
            "family": dict(Counter(r["task_family"] for r in records)),
            "depth": dict(sorted(Counter(r["depth"] for r in records).items())),
            "question_type": dict(Counter(
                r["question_variant"] if r["question_variant"] == "bound"
                else r["question_type"] for r in records)),
            "domain": dict(Counter(r["domain"] for r in records)),
        }
    return table


def template_variety(records: Sequence[dict]) -> dict:
    """Distinct surface configurations actually realized (§11.5).

    The polarity pattern is the sequence of stated directions across a
    problem's premises -- ``('>', '<', '>')`` and so on.  Counting distinct
    patterns is what distinguishes "the generator can flip polarity" from "the
    generator did flip polarity, in many different combinations".

    Args:
        records: All records for one language.

    Returns:
        Distinct counts and the realized combination total.
    """
    polarity = {tuple(p["rel"] for p in r["premises"]) for r in records}
    combos = {(r["task_family"], r["domain"], r["question_type"],
               r["question_variant"], r["depth"]) for r in records}
    return {
        "domains": len({r["domain"] for r in records}),
        "families": len({r["task_family"] for r in records}),
        "question_types": len({(r["question_type"], r["question_variant"])
                               for r in records}),
        "depths": sorted({r["depth"] for r in records}),
        "polarity_patterns": len(polarity),
        "family_domain_question_depth_combinations": len(combos),
    }


def collect_samples(by_slice: dict[str, list[dict]], per_group: int = 2
                    ) -> list[dict]:
    """Two rendered examples per (family x question type), for eyeballing (§11.6).

    Args:
        by_slice: Records grouped by slice tag.
        per_group: How many examples per group.

    Returns:
        Sample records, each carrying the group it illustrates.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for tag, records in by_slice.items():
        for record in records:
            key = (record["task_family"], record["question_type"],
                   record["question_variant"])
            if len(grouped[key]) < per_group:
                grouped[key].append(record)
    return [{"group": list(key), "record": r}
            for key, group in sorted(grouped.items()) for r in group]


def collect_agreement_samples(by_slice: dict[str, list[dict]],
                              genders: dict[str, str]) -> list[dict]:
    """Samples chosen to make the revision's two fixes visible.

    The general sampler in :func:`collect_samples` covers every family and
    question type, but it has no reason to pick a feminine-named entity or a
    wealth problem, and those are exactly what a reviewer needs to see to
    confirm Fix A and Fix B landed.  This picks, deliberately:

    * one record per domain whose FINAL sentence has a feminine subject, so the
      adjective agreement is on display rather than merely claimed;
    * one relational and one numeric-anchor wealth record, so both the
      comparative phrasing and the possession phrasing can be read.

    Args:
        by_slice: Records grouped by slice tag.
        genders: Name to gender, pooled across both name pools.

    Returns:
        Chosen samples, each tagged with what it illustrates.
    """
    def final_subject(record: dict) -> str | None:
        """The entity the final natural-language sentence agrees with."""
        query = record["query"]
        if query["kind"] == "extremum":
            return record["answer"]
        if query["kind"] == "pairwise":
            return query["a"]
        return None

    picked: dict[str, dict] = {}
    for tag in sorted(by_slice):
        for record in by_slice[tag]:
            domain = record["domain"]
            subject = final_subject(record)
            if subject and genders.get(subject) == "f":
                picked.setdefault(f"{domain} — feminine subject", record)
            if domain == "wealth":
                key = ("wealth — numeric anchor" if record["numeric_anchor"]
                       else "wealth — relational")
                picked.setdefault(key, record)
    return [{"group": [label], "record": record}
            for label, record in sorted(picked.items())]


def domain_leakage_report(by_slice: dict[str, list[dict]],
                          ood_domains: Sequence[str]) -> dict:
    """Confirm no OOD-only domain reached a training file (§4, §12).

    Args:
        by_slice: Records grouped by slice tag.
        ood_domains: Domains flagged ``ood_only`` in the language config.

    Returns:
        A check-result dict.
    """
    offenders = {
        tag: sorted({r["domain"] for r in records} & set(ood_domains))
        for tag, records in by_slice.items()
        if records and not records[0]["split"].startswith("test")
    }
    offenders = {tag: doms for tag, doms in offenders.items() if doms}
    return checks.check_result(
        "G6.ood_domain", not offenders,
        f"OOD-only domains {list(ood_domains)} absent from every training file",
        {"ood_domains": list(ood_domains), "offenders": offenders})


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def load_tokenizer(config: dict, root: Path):
    """Open one language's Phase 2 BPE tokenizer, read-only.

    Args:
        config: Language config, holding the repo-relative tokenizer path.
        root: Repository root.

    Returns:
        A loaded ``SentencePieceProcessor``.

    Raises:
        FileNotFoundError: If the tokenizer is missing, which would silently
            turn G5 into an unmeasured claim if it were tolerated.
    """
    import sentencepiece as spm

    path = (root / config["tokenizer"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    processor = spm.SentencePieceProcessor()
    processor.load(str(path))
    return processor


def check_determinism(config: dict, build_config: dict,
                      digests: dict[str, str]) -> dict:
    """G7: a second independent build reproduces every slice byte for byte.

    The whole language is regenerated from the master seed with a fresh prompt
    registry and compared by digest.  Re-hashing the files that were just
    written would only prove that a file is equal to itself; regenerating
    proves the generator is a pure function of its seed.

    Args:
        config: Language config.
        build_config: The build plan.
        digests: Slice tag to digest from the first build.

    Returns:
        A check-result dict.
    """
    _, by_slice, _ = build_language(config, build_config)
    rebuilt = {tag: digest(serialize(records))
               for tag, records in by_slice.items()}
    mismatches = sorted(tag for tag in digests
                        if digests[tag] != rebuilt.get(tag))
    return checks.check_result(
        "G7", not mismatches,
        f"{len(digests) - len(mismatches)}/{len(digests)} slices reproduced "
        f"byte-identically from the same seed",
        {"slices": len(digests), "mismatches": mismatches,
         "digests": digests})


def run_checks(config: dict, build_config: dict, records: Sequence[dict],
               by_slice: dict[str, list[dict]], digests: dict[str, str],
               root: Path) -> dict:
    """Run G1-G8 for one language and collect the reported numbers.

    Args:
        config: Language config.
        build_config: The build plan.
        records: Every record for this language.
        by_slice: The same records grouped by slice.
        digests: Per-slice digests from this build.
        root: Repository root, for locating the tokenizer.

    Returns:
        ``{"checks": [...], "name_lengths": {...}}``.
    """
    tokenizer = load_tokenizer(config, root)
    train_names = generator.load_names(config["names"]["train"])
    test_names = generator.load_names(config["names"]["test"])
    ood_domains = [name for name, block in config["domains"].items()
                   if block.get("ood_only")]

    results = [
        checks.check_well_posed(records),
        checks.check_solver_agreement(records, config),
        checks.check_non_adjacency(records),
        checks.check_nonconsecutive_coverage(
            records, build_config["nonconsecutive"]["min_depth"]),
        checks.measure_lengths(records, tokenizer, CONTEXT_LEN, WARN_AT),
        checks.check_pool_disjointness(train_names, test_names),
        checks.check_name_length_gap(train_names, test_names, tokenizer),
        checks.check_prompt_overlap(records),
        checks.check_label_balance(records, config),
        checks.check_extremum_gender_neutral(records, config),
        checks.check_name_answer_independence(records),
        domain_leakage_report(by_slice, ood_domains),
        check_determinism(config, build_config, digests),
        checks.check_answer_extraction(records, config,
                                       generator.extract_answer),
    ]
    return {
        "checks": results,
        "name_lengths": checks.measure_name_lengths(
            train_names, test_names, tokenizer),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def process_language(config_stem: str, build_config: dict,
                     config_dir: Path, out_root: Path, root: Path,
                     code: str | None = None) -> dict:
    """Generate, write and verify one language end to end.

    Args:
        code: ``"hi"``.
        build_config: The build plan.
        config_dir: Directory holding the language configs.
        out_root: ``artifacts/finetune_data/part1``; the language's own directory is created
            under it.
        root: Repository root.

    Returns:
        The full summary for this language.
    """
    code = code or config_stem
    config = generator.load_language_config(config_dir / f"{config_stem}.yaml")
    print(f"[{code}] generating {config['language']} ...", flush=True)
    records, by_slice, slice_stats = build_language(config, build_config)

    out_dir = out_root / code
    out_dir.mkdir(parents=True, exist_ok=True)
    digests, files = {}, {}
    for tag, slice_records in by_slice.items():
        text = serialize(slice_records)
        name = output_name(tag, build_config["slices"][tag])
        (out_dir / name).write_text(text, encoding="utf-8")
        digests[tag] = digest(text)
        files[tag] = relative_path(out_dir / name, root)
        print(f"[{code}]   {name:28s} {len(slice_records):6d} records",
              flush=True)

    print(f"[{code}] verifying ...", flush=True)
    verification = run_checks(config, build_config, records, by_slice,
                              digests, root)

    # Pooled across both pools: a record's names come from one pool, but the
    # lookup is only ever used for display.
    genders = {**generator.load_name_genders(config["names"]["train"]),
               **generator.load_name_genders(config["names"]["test"])}

    return {
        "language": config["language"],
        "code": code,
        "tokenizer": config["tokenizer"],
        "files": files,
        "digests": digests,
        "slice_stats": slice_stats,
        "counts": count_table(by_slice),
        "variety": template_variety(records),
        "samples": collect_samples(by_slice),
        "agreement_samples": collect_agreement_samples(by_slice, genders),
        "total_records": len(records),
        **verification,
    }


def main() -> None:
    """Build both datasets, verify them, and write RESULTS_part1.md."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hi",),
                        default="hi", help="which dataset(s) to build")
    parser.add_argument("--root", type=Path, default=_HERE.parent.parent,
                        help="repository root")
    parser.add_argument("--config-dir", type=Path, default=_HERE / "configs")
    parser.add_argument("--build-config", default="build.yaml",
                        help="build plan inside --config-dir")
    parser.add_argument("--config-suffix", default="",
                        help="language config suffix, e.g. '_perf'")
    parser.add_argument("--out", type=Path, default=None,
                        help="dataset output root (default artifacts/finetune_data/part1)")
    parser.add_argument("--summary", type=Path, default=None,
                        help="machine-readable summary JSON")
    parser.add_argument("--results", type=Path, default=None,
                        help="RESULTS_part1.md destination")
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = (args.out or root / "artifacts" / "finetune_data" / "part1").resolve()
    summary_path = (args.summary
                    or root / "reports" / "gpt_finetune" / "finetune_data" / "summary_part1.json")
    results_path = args.results or root / "reports" / "gpt_finetune" / "finetune_data" / "RESULTS_part1.md"

    build_config = generator.load_build_config(
        args.config_dir / args.build_config)
    codes = ("hi",)

    summaries = [process_language(code + args.config_suffix, build_config,
                                  args.config_dir, out_root, root,
                                  code=code)
                 for code in codes]

    summary = {"build_config": build_config, "languages": summaries}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    import report
    report.write_results(summary, results_path, root)

    failed = [(s["code"], c["check"]) for s in summaries
              for c in s["checks"] if c["status"] != checks.PASS]
    print()
    print(report.render_check_table(summaries))
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    total = sum(s["total_records"] for s in summaries)
    print(f"All checks passed. {total} records across "
          f"{len(summaries)} languages.")
    print(f"Dataset:  {out_root}")
    print(f"Summary:  {summary_path}")
    print(f"Results:  {results_path}")


if __name__ == "__main__":
    main()
