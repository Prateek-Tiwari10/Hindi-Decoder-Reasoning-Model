"""Build the four nested training subsets for the Phase 3 scaling study.

    python -m data.reasoning.make_subsets

WHY PREFIXES OF ONE SHUFFLE, RATHER THAN FOUR SAMPLES
-----------------------------------------------------
The scaling curve has to mean "the same data plus more", not "four different
datasets".  Four independent draws would confound size with composition: a
10k draw and a 50k draw could differ in family mix, depth mix or which names
they happen to cover, and any accuracy difference would then be unattributable.
One seeded shuffle followed by prefix slicing makes 10k a literal subset of
30k, 30k of 50k, and 50k of 60k -- the nesting is a property of the
construction, and check S1 verifies it rather than assuming it.

One fixed shuffle seed defines the order, so the subsets are reproducible
byte for byte and strictly nested.

NOTHING IS REGENERATED HERE.  ``train.jsonl`` is read, its lines are reordered,
and prefixes are written out.  Every record in every subset is a byte-identical
line of the already-validated Perf build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

#: Shuffle seed.  Recorded in the manifest and in RESULTS so the subsets can be
#: rebuilt exactly; fixed on purpose (see module docstring).
SUBSET_SEED = 20260912

#: The four points on the scaling curve, smallest first.  Order matters: the
#: nesting assertion walks adjacent pairs.
SIZES = (10_000, 30_000, 50_000, 60_000)

#: Per-name repetition count at which memorising the name inventory became the
#: cheaper strategy than copying from the prompt (measured in Part 3A).  Every
#: subset is reported as a multiple below this.
MEMORIZATION_REGIME = 455

#: A subset whose worst name exceeds this many appearances would mean the
#: shuffle clustered that name rather than spreading it.
MAX_NAME_REPS_FLAG = 120


def read_lines(path: Path) -> list[str]:
    """Read a JSONL file as raw lines, newline stripped.

    Lines are kept as text rather than parsed and re-dumped: re-serializing
    would risk changing key order or spacing, and the point of the subsets is
    that every line is byte-identical to the Perf build it came from.

    Args:
        path: The JSONL file.

    Returns:
        One string per non-empty line.
    """
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_pool_names(path: Path) -> list[str]:
    """Read a name pool file as a list of names.

    Args:
        path: Path to a ``*_names_*.txt`` pool.

    Returns:
        The names, comments and blank lines skipped.
    """
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split("\t")[0].strip())
    return names


def recurrence_stats(records: Sequence[dict], pool: Sequence[str]) -> dict:
    """How often each pool name is reused across one subset.

    The statistics are taken over the WHOLE pool, not over the names that
    happen to appear: a name the subset never uses is a zero, and averaging
    only over the names present would quietly inflate the mean and hide exactly
    the thin coverage that a small subset is expected to have.

    Args:
        records: The subset's parsed records.
        pool: Every name in that language's train pool.

    Returns:
        Slot total, pool size, coverage, and min / median / mean / max
        appearances per pool name.
    """
    counts = Counter()
    for record in records:
        for name in record["true_order"]:
            counts[name] += 1
    per_name = [counts.get(name, 0) for name in pool]
    return {
        "name_slots": sum(per_name),
        "pool_size": len(pool),
        "names_used": sum(1 for c in per_name if c > 0),
        "names_unused": sum(1 for c in per_name if c == 0),
        "min": min(per_name),
        "median": statistics.median(per_name),
        "mean": round(statistics.fmean(per_name), 2),
        "max": max(per_name),
        "headroom_vs_memorization": round(
            MEMORIZATION_REGIME / statistics.fmean(per_name), 1),
    }


def composition(records: Sequence[dict]) -> dict:
    """Family / depth / domain breakdown of one subset, with percentages.

    Args:
        records: The subset's parsed records.

    Returns:
        Counts and percentages for each of the three axes.
    """
    total = len(records)

    def breakdown(field: str) -> dict:
        counter = Counter(record[field] for record in records)
        return {
            "counts": {str(k): v for k, v in sorted(counter.items(),
                                                    key=lambda kv: str(kv[0]))},
            "pct": {str(k): round(100.0 * v / total, 2)
                    for k, v in sorted(counter.items(),
                                       key=lambda kv: str(kv[0]))},
        }

    return {"family": breakdown("task_family"), "depth": breakdown("depth"),
            "domain": breakdown("domain")}


def build_language(code: str, source: Path, out_dir: Path, pool_path: Path
                   ) -> dict:
    """Shuffle one language's training file and write its four prefixes.

    Args:
        code: ``"hi"``.
        source: That language's ``train.jsonl`` from the Perf build.
        out_dir: Destination directory for ``train_<size>.jsonl``.
        pool_path: That language's train name pool, for recurrence stats.

    Returns:
        A per-size manifest entry, plus the nesting check result.

    Raises:
        ValueError: If the source file does not hold exactly 60,000 records,
            which would mean it is not the Perf training set.
    """
    lines = read_lines(source)
    if len(lines) != max(SIZES):
        raise ValueError(
            f"{source}: expected {max(SIZES)} records, found {len(lines)}")

    # Shuffle a COPY: the source file on disk is never touched.
    shuffled = list(lines)
    random.Random(SUBSET_SEED).shuffle(shuffled)

    pool = load_pool_names(pool_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets, id_sets = {}, {}
    for size in SIZES:
        prefix = shuffled[:size]
        records = [json.loads(line) for line in prefix]
        text = "".join(line + "\n" for line in prefix)
        name = f"train_{size // 1000}k.jsonl"
        (out_dir / name).write_text(text, encoding="utf-8")
        id_sets[size] = {record["id"] for record in records}
        subsets[size] = {
            "file": name,
            "records": len(records),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "composition": composition(records),
            "recurrence": recurrence_stats(records, pool),
        }
        print(f"[{code}]   {name:18s} {len(records):6d} records  "
              f"mean reps/name {subsets[size]['recurrence']['mean']:5.2f}  "
              f"max {subsets[size]['recurrence']['max']}")

    # S1: nesting is asserted, not assumed.
    nesting = []
    for smaller, larger in zip(SIZES, SIZES[1:]):
        contained = id_sets[smaller] <= id_sets[larger]
        nesting.append({"smaller": smaller, "larger": larger,
                        "smaller_ids": len(id_sets[smaller]),
                        "larger_ids": len(id_sets[larger]),
                        "is_subset": contained})
    return {"subsets": subsets, "nesting": nesting,
            "all_nested": all(step["is_subset"] for step in nesting)}


def main() -> None:
    """Build the Hindi subsets and print the S1 verdict."""
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path,
                        default=root / "artifacts" / "finetune_data")
    parser.add_argument("--out", type=Path,
                        help="subset root (default <data-root>/train)")
    parser.add_argument("--manifest", type=Path,
                        default=here / "summary_subsets.json")
    args = parser.parse_args()

    out_root = args.out or args.data_root / "train"
    result = {"subset_seed": SUBSET_SEED, "sizes": list(SIZES),
              "languages": {}}
    for code in ("hi",):
        print(f"[{code}] slicing {args.data_root / code / 'train.jsonl'} ...")
        result["languages"][code] = build_language(
            code, args.data_root / code / "train.jsonl", out_root / code,
            here / "vocab" / f"{code}_names_train_perf.txt")

    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [code for code, block in result["languages"].items()
                if not block["all_nested"]]
    print()
    for code, block in result["languages"].items():
        verdict = "PASS" if block["all_nested"] else "FAIL"
        print(f"S1 nesting [{code}]: {verdict} — " + ", ".join(
            f"{s['smaller']//1000}k ⊂ {s['larger']//1000}k"
            for s in block["nesting"]))
    print(f"Subsets:  {out_root}")
    print(f"Manifest: {args.manifest}")
    if failures:
        raise SystemExit(f"FAILED: nesting broken for {failures}")


if __name__ == "__main__":
    main()
