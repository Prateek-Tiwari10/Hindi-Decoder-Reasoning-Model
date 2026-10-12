"""Assemble the Phase 3 scaling-study data tree and run its self-checks.

    python -m data.reasoning.build_scaling_tree --xdom-staging <dir>

This module NEVER generates a record.  ``make_subsets.py`` has already written
the four nested training subsets, and ``run.py --build-config build_xdom.yaml``
has already written the cross-domain slices; this places both into the layout
Part B will read, carries the validated Perf files across by sha256
verification, runs the study-specific checks S1-S8, and emits
``manifest_scaling.json`` plus the numbers ``RESULTS_scaling_data.md`` renders.

The carried-over files are verified, never regenerated.  A mismatch against the
Perf RESULTS section 9 digest is a hard stop: it would mean the corpus this
study is scaling is not the corpus that was validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import checks                                        # noqa: E402
import generator                                     # noqa: E402

CONTEXT_LEN = 512

#: sha256 (first 16) of every carried-over file, as recorded in
#: the course run's data-provenance report (sha256 table).  Hard-coded so the carry-over is an
#: assertion against a written record, not against whatever is on disk now.
PERF_DIGESTS = {
    ("hi", "train.jsonl"): "5b649e5d526ee62b",
    ("hi", "val.jsonl"): "350b16f22a628e9d",
    ("hi", "val_heldout.jsonl"): "2d16d0725966b3d1",
    ("hi", "test_rel_id_d23.jsonl"): "bd42971a9f0493cf",
    ("hi", "test_num_id_d23.jsonl"): "397bed0b74af8a6e",
    ("hi", "test_mix_id_d23.jsonl"): "a07b57734c41708c",
    ("hi", "test_rel_ood_d4.jsonl"): "d98727150c2f5c65",
    ("hi", "test_rel_ood_d5.jsonl"): "b7fa4bfcc82afa50",
    ("hi", "test_rel_ood_d6.jsonl"): "7d8db9b98b0e669c",
}

#: Carried-over file -> its name in the new tree.
VALIDATION_FILES = {"val.jsonl": "val_mix.jsonl",
                    "val_heldout.jsonl": "val_heldout.jsonl"}
IN_DOMAIN_FILES = {f"test_{tag}.jsonl": f"{tag}.jsonl" for tag in
                   ("rel_id_d23", "num_id_d23", "mix_id_d23",
                    "rel_ood_d4", "rel_ood_d5", "rel_ood_d6")}

XDOM_DOMAINS = ("speed", "strength", "diligence", "temperature")
XDOM_SLICES = ("id_d23", "ood_d4", "ood_d5", "ood_d6")
SUBSET_SIZES = (10_000, 30_000, 50_000, 60_000)

#: S3: a per-name maximum above this would mean the shuffle clustered a name.
MAX_NAME_REPS_FLAG = 120
#: S2: a subset's family mixture may drift this far from the full set, in pp.
MIXTURE_TOLERANCE_PP = 2.0


def sha256_file(path: Path) -> str:
    """Full sha256 of a file's bytes.

    Args:
        path: The file.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict]:
    """Parse a JSONL file.

    Args:
        path: The file.

    Returns:
        One dict per line.
    """
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line]


def copy_verified(source: Path, destination: Path, expected16: str) -> dict:
    """Copy one carried-over file, asserting its digest before and after.

    Args:
        source: The Perf build file.
        destination: Where it belongs in the new tree.
        expected16: First 16 hex chars recorded in Perf RESULTS section 9.

    Returns:
        A record of the copy, including both digests.

    Raises:
        SystemExit: If the source does not match the recorded digest, or the
            copy does not match the source.
    """
    digest = sha256_file(source)
    if digest[:16] != expected16:
        raise SystemExit(
            f"STOP: {source} sha256 {digest[:16]} != recorded {expected16}. "
            f"The Perf corpus on disk is not the one RESULTS section 9 "
            f"validated; not regenerating, not proceeding.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied = sha256_file(destination)
    if copied != digest:
        raise SystemExit(f"STOP: copy of {source} does not match its source")
    return {"source": str(source), "file": str(destination),
            "sha256": digest, "records": sum(
                1 for line in destination.read_text(encoding="utf-8").splitlines()
                if line)}


def slice_composition(records: Sequence[dict]) -> dict:
    """Depth and question-type mix of one slice.

    Args:
        records: The slice's records.

    Returns:
        Counts by depth, question type, domain and family.
    """
    return {
        "records": len(records),
        "depth": dict(sorted(Counter(r["depth"] for r in records).items())),
        "question_type": dict(sorted(Counter(
            r["question_type"] for r in records).items())),
        "domain": dict(sorted(Counter(r["domain"] for r in records).items())),
        "family": dict(sorted(Counter(r["task_family"] for r in records).items())),
    }


def sequence_lengths(records: Sequence[dict], tokenizer) -> dict:
    """S6: token lengths of prompt+completion for one slice.

    Args:
        records: The slice's records.
        tokenizer: That language's 16K SentencePiece model, read-only.

    Returns:
        max, p99 and mean token counts, and the pass verdict against the
        512-token context.
    """
    lengths = sorted(len(tokenizer.encode(r["prompt"] + r["completion"]))
                     for r in records)
    index = min(len(lengths) - 1, int(0.99 * len(lengths)))
    return {"max": lengths[-1], "p99": lengths[index],
            "mean": round(sum(lengths) / len(lengths), 1),
            "fits_context": lengths[-1] < CONTEXT_LEN}


def names_in(records: Sequence[dict]) -> set[str]:
    """Every entity name appearing in a set of records.

    Args:
        records: Records to scan.

    Returns:
        The set of names used.
    """
    return {name for record in records for name in record["true_order"]}


def main() -> None:
    """Assemble the tree, run S1-S8, and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    root = _HERE.parent.parent
    parser.add_argument("--data-root", type=Path,
                        default=root / "artifacts" / "finetune_data")
    parser.add_argument("--xdom-staging", type=Path, required=True,
                        help="where run.py --build-config build_xdom.yaml wrote")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report-json", type=Path,
                        default=_HERE / "summary_scaling.json")
    args = parser.parse_args()

    data_root = args.data_root
    manifest_path = args.manifest or data_root / "manifest_scaling.json"
    subset_summary = json.loads(
        (_HERE / "summary_subsets.json").read_text(encoding="utf-8"))

    import sentencepiece as spm

    manifest: dict = {"subset_seed": subset_summary["subset_seed"],
                      "master_seed": 20260910,
                      "sizes": list(SUBSET_SIZES),
                      "unseen_domains": list(XDOM_DOMAINS),
                      "languages": {}}
    study: dict = {"carried_over": [], "checks": {}, "slices": {},
                   "subsets": subset_summary["languages"]}

    for code in ("hi",):
        config = generator.load_language_config(_HERE / "configs"
                                                / f"{code}_perf.yaml")
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str((root / config["tokenizer"]).resolve()))

        lang_manifest: dict = {"train": {}, "validation": {}, "test": {}}

        # The subsets are cut from this file, so it is verified against the
        # Perf RESULTS section 9 digest as strictly as the copied files are --
        # otherwise the scaling study could be built on an unvalidated corpus.
        train_source = data_root / code / "train.jsonl"
        train_digest = sha256_file(train_source)
        if train_digest[:16] != PERF_DIGESTS[(code, "train.jsonl")]:
            raise SystemExit(
                f"STOP: {train_source} sha256 {train_digest[:16]} != recorded "
                f"{PERF_DIGESTS[(code, 'train.jsonl')]}. The subsets would be "
                f"cut from an unvalidated corpus; not proceeding.")
        study["train_source_verified"] = study.get("train_source_verified", [])
        study["train_source_verified"].append(
            {"language": code, "file": str(train_source.relative_to(root)),
             "sha256": train_digest, "matches_perf_results": True})

        # ---- carried-over validation and in-domain test files -------------
        for src_name, dst_name in VALIDATION_FILES.items():
            entry = copy_verified(data_root / code / src_name,
                                  data_root / "validation" / code / dst_name,
                                  PERF_DIGESTS[(code, src_name)])
            entry.update({"language": code, "role": "validation",
                          "slice": dst_name[:-6]})
            study["carried_over"].append(entry)
            lang_manifest["validation"][dst_name[:-6]] = {
                "path": str(Path(entry["file"]).relative_to(root)),
                "records": entry["records"], "sha256": entry["sha256"]}

        for src_name, dst_name in IN_DOMAIN_FILES.items():
            entry = copy_verified(
                data_root / code / src_name,
                data_root / "test" / code / "in_domain" / dst_name,
                PERF_DIGESTS[(code, src_name)])
            entry.update({"language": code, "role": "test/in_domain",
                          "slice": dst_name[:-6]})
            study["carried_over"].append(entry)
            records = load_records(Path(entry["file"]))
            lang_manifest["test"][f"in_domain/{dst_name[:-6]}"] = {
                "path": str(Path(entry["file"]).relative_to(root)),
                "records": entry["records"], "sha256": entry["sha256"],
                "family": sorted({r["task_family"] for r in records}),
                "depths": sorted({r["depth"] for r in records}),
                "domains": sorted({r["domain"] for r in records}),
                "unseen_domain": False}

        # ---- training subsets (already written by make_subsets.py) --------
        for size in SUBSET_SIZES:
            name = f"train_{size // 1000}k.jsonl"
            path = data_root / "train" / code / name
            block = subset_summary["languages"][code]["subsets"][str(size)]
            lang_manifest["train"][f"{size // 1000}k"] = {
                "path": str(path.relative_to(root)),
                "records": block["records"], "sha256": block["sha256"],
                "recurrence": block["recurrence"],
                "family_pct": block["composition"]["family"]["pct"],
                "depth_pct": block["composition"]["depth"]["pct"],
                "domain_pct": block["composition"]["domain"]["pct"]}

        # ---- cross-domain slices ------------------------------------------
        for domain in XDOM_DOMAINS:
            for suffix in XDOM_SLICES:
                tag = f"xdom_{domain}_{suffix}"
                source = args.xdom_staging / code / f"{tag}.jsonl"
                destination = (data_root / "test" / code / "cross_domain"
                               / domain / f"{tag}.jsonl")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                records = load_records(destination)
                digest = sha256_file(destination)
                if digest != sha256_file(source):
                    raise SystemExit(f"STOP: copy of {source} differs")
                composition = slice_composition(records)
                lengths = sequence_lengths(records, tokenizer)
                study["slices"][f"{code}/{tag}"] = {
                    "language": code, "domain": domain, "slice": tag,
                    "path": str(destination.relative_to(root)),
                    "sha256": digest, **composition, "lengths": lengths,
                    "G1": checks.check_well_posed(records),
                    "G2": checks.check_solver_agreement(records, config),
                    "G8": checks.check_answer_extraction(
                        records, config, generator.extract_answer),
                }
                lang_manifest["test"][f"cross_domain/{domain}/{tag}"] = {
                    "path": str(destination.relative_to(root)),
                    "records": len(records), "sha256": digest,
                    "family": ["relation"],
                    "depths": sorted(composition["depth"]),
                    "domains": [domain], "unseen_domain": True}

        manifest["languages"][code] = lang_manifest

    # ---- study-specific checks S2 - S5 ------------------------------------
    s2, s3, s4, s5 = [], [], [], []
    for code in ("hi",):
        full = subset_summary["languages"][code]["subsets"][str(max(SUBSET_SIZES))]
        full_family = full["composition"]["family"]["pct"]
        for size in SUBSET_SIZES:
            block = subset_summary["languages"][code]["subsets"][str(size)]
            drift = {family: round(block["composition"]["family"]["pct"][family]
                                   - full_family[family], 2)
                     for family in full_family}
            s2.append({"language": code, "size": size, "pct":
                       block["composition"]["family"]["pct"], "drift_pp": drift,
                       "pass": max(abs(v) for v in drift.values())
                       <= MIXTURE_TOLERANCE_PP})
            s3.append({"language": code, "size": size,
                       **block["recurrence"],
                       "pass": block["recurrence"]["max"] <= MAX_NAME_REPS_FLAG})

        # S4 / S5 read the subsets back off disk rather than trusting the
        # manifest: the assertion has to be about the files Part B will train on.
        train_names: set[str] = set()
        for size in SUBSET_SIZES:
            records = load_records(
                args.data_root / "train" / code / f"train_{size // 1000}k.jsonl")
            counts = {domain: sum(1 for r in records if r["domain"] == domain)
                      for domain in XDOM_DOMAINS}
            s4.append({"language": code, "size": size, "counts": counts,
                       "pass": all(v == 0 for v in counts.values())})
            train_names |= names_in(records)

        for key, block in study["slices"].items():
            if not key.startswith(f"{code}/"):
                continue
            slice_names = names_in(load_records(root / block["path"]))
            shared = sorted(slice_names & train_names)
            s5.append({"language": code, "slice": block["slice"],
                       "slice_names": len(slice_names),
                       "train_names": len(train_names),
                       "shared": len(shared), "examples": shared[:5],
                       "pass": not shared})

    study["checks"] = {
        "S1": [{"language": code,
                "steps": subset_summary["languages"][code]["nesting"],
                "pass": subset_summary["languages"][code]["all_nested"]}
               for code in ("hi",)],
        "S2": s2, "S3": s3, "S4": s4, "S5": s5,
        "S6": [{"language": b["language"], "slice": b["slice"], **b["lengths"],
                "pass": b["lengths"]["fits_context"]}
               for b in study["slices"].values()],
        "S7": [{"language": b["language"], "slice": b["slice"],
                "G1": b["G1"]["status"], "G2": b["G2"]["status"],
                "pass": b["G1"]["status"] == checks.PASS
                        and b["G2"]["status"] == checks.PASS}
               for b in study["slices"].values()],
        "S8": [{"language": b["language"], "slice": b["slice"],
                "G8": b["G8"]["status"], "detail": b["G8"]["detail"],
                "pass": b["G8"]["status"] == checks.PASS}
               for b in study["slices"].values()],
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    args.report_json.write_text(
        json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    failed = []
    for name, rows in study["checks"].items():
        bad = [r for r in rows if not r["pass"]]
        verdict = "PASS" if not bad else f"FAIL ({len(bad)})"
        print(f"{name}: {verdict}  over {len(rows)} rows")
        if bad:
            failed.append(name)
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Study:    {args.report_json}")
    if failed:
        raise SystemExit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
