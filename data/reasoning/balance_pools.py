"""Re-partition the name pools so train and test match in subword length.

WHY THIS EXISTS
---------------
The first hand-authored split put the more familiar, shorter names in the
train pool and the rarer, longer ones in the test pool.  Measured with the
Phase 2 16K tokenizers that came to +0.43 subword tokens per name for Hindi --
test names were a third longer than train names in the units the model
actually consumes.

That is a confound, not a leak.  Test accuracy is supposed to drop only
because the names are unseen; if the test names are also systematically harder
to tokenize, a drop cannot be attributed to either cause.  §9.5 asks for the
gap to be measured and flagged; measuring it and then removing it is strictly
better, and costs nothing but a re-partition.

WHAT IT DOES
------------
Pools both name lists for a language, buckets the names by their subword
length under that language's own tokenizer, and re-splits each bucket into
train and test in the original 220:70 proportion using largest-remainder
apportionment.  Pool sizes, disjointness and the names themselves are
unchanged -- only which side of the split each name falls on.  Seeded, so the
partition is reproducible.

Run once; the committed pool files are its output.

    python -m data.reasoning.balance_pools
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent

#: Header written above each regenerated pool file, per language and side.
HEADERS = {
    ("hi", "train"): "Hindi given-name pool -- TRAIN side.",
    ("hi", "test"): "Hindi given-name pool -- TEST side.",
}

NOTE = """#
# Format: <name>\\t<gender>   where gender is `m` or `f`.
# Lines beginning with `#` and blank lines are ignored.
#
# The gender tag is LOAD-BEARING as of revision Fix B: predicate adjectives
# agree with the subject entity's gender, and this tag is where that gender
# comes from.  Gender is a property of the sampled name and not a new random
# draw, so seed-determinism (check G7) is unaffected.
#
# THIS SPLIT IS STRATIFIED BY SUBWORD LENGTH, not hand-assigned.  The train and
# test pools are matched on mean tokens per name under this language's Phase 2
# 16K BPE tokenizer, so that a drop in test accuracy cannot be attributed to
# test names simply being harder to tokenize.  Regenerate with
# `python -m data.reasoning.balance_pools`.
#
# The two pools are DISJOINT -- the primary leakage control (§9.1), asserted by
# check G6.  Overlap with the OTHER language's pools is irrelevant: the two
# models share no data, vocabulary or weights.
#
# HUMAN REVIEW REQUESTED (§9.1): a native speaker should confirm that none of
# these read as rare, regional-only, or as imports from the other language.
"""


def read_pool(path: Path) -> list[tuple[str, str]]:
    """Read a pool file as (name, gender) pairs.

    Args:
        path: Pool file.

    Returns:
        The entries in file order.
    """
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entries.append((parts[0], parts[1] if len(parts) > 1 else "m"))
    return entries


def write_pool(path: Path, code: str, side: str,
               entries: list[tuple[str, str]]) -> None:
    """Write a pool file with its header.

    Args:
        path: Destination.
        code: Language code, for the header.
        side: ``"train"`` or ``"test"``.
        entries: (name, gender) pairs, already ordered.
    """
    lines = [f"# {HEADERS[(code, side)]}", NOTE.rstrip(), ""]
    lines += [f"{name}\t{gender}" for name, gender in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apportion(bucket_sizes: dict[int, int], total_test: int) -> dict[int, int]:
    """Largest-remainder apportionment of the test quota across buckets.

    Proportional rounding alone would not sum to the required total; the
    largest-remainder method distributes the leftover seats to the buckets with
    the largest fractional parts, which keeps every bucket as close to its
    proportional share as an integer allows.

    Args:
        bucket_sizes: Bucket key to number of names in it.
        total_test: How many names must land in the test pool overall.

    Returns:
        Bucket key to its integer test quota.
    """
    total = sum(bucket_sizes.values())
    exact = {k: n * total_test / total for k, n in bucket_sizes.items()}
    quota = {k: int(v) for k, v in exact.items()}
    remaining = total_test - sum(quota.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))
    for key in order[:remaining]:
        quota[key] += 1
    return quota


def balance(code: str, vocab_dir: Path, config_dir: Path, root: Path,
            seed: int = 20260910) -> dict:
    """Re-partition one language's pools, matched on subword length.

    Args:
        code: ``"hi"``.
        vocab_dir: Directory holding the pool files.
        config_dir: Directory holding the language configs.
        root: Repository root, for resolving the tokenizer path.
        seed: Seed for the within-bucket shuffle.

    Returns:
        Before/after mean subword lengths and the resulting pool sizes.
    """
    import sentencepiece as spm

    config = yaml.safe_load((config_dir / f"{code}.yaml").read_text(
        encoding="utf-8"))
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str((root / config["tokenizer"]).resolve()))

    train_path = vocab_dir / f"{code}_names_train.txt"
    test_path = vocab_dir / f"{code}_names_test.txt"
    train_before, test_before = read_pool(train_path), read_pool(test_path)
    n_test = len(test_before)

    everyone = train_before + test_before
    lengths = {name: len(tokenizer.encode(name)) for name, _ in everyone}

    buckets: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for entry in everyone:
        buckets[lengths[entry[0]]].append(entry)

    quota = apportion({k: len(v) for k, v in buckets.items()}, n_test)
    rng = random.Random(seed)
    train_after: list[tuple[str, str]] = []
    test_after: list[tuple[str, str]] = []
    for key in sorted(buckets):
        members = sorted(buckets[key])
        rng.shuffle(members)
        test_after += members[:quota[key]]
        train_after += members[quota[key]:]

    # Order each pool by gender then name so the files stay readable.
    train_after.sort(key=lambda e: (e[1], e[0]))
    test_after.sort(key=lambda e: (e[1], e[0]))
    write_pool(train_path, code, "train", train_after)
    write_pool(test_path, code, "test", test_after)

    def mean_of(entries: list[tuple[str, str]]) -> float:
        return round(statistics.fmean(lengths[n] for n, _ in entries), 4)

    return {
        "code": code,
        "before": {"train": mean_of(train_before), "test": mean_of(test_before),
                   "gap": round(mean_of(test_before) - mean_of(train_before), 4)},
        "after": {"train": mean_of(train_after), "test": mean_of(test_after),
                  "gap": round(mean_of(test_after) - mean_of(train_after), 4)},
        "sizes": {"train": len(train_after), "test": len(test_after)},
        "overlap": len({n for n, _ in train_after} & {n for n, _ in test_after}),
    }


def main() -> None:
    """Re-partition the Hindi name pools and print the before/after gaps."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=_HERE.parent.parent)
    parser.add_argument("--vocab-dir", type=Path, default=_HERE / "vocab")
    parser.add_argument("--config-dir", type=Path, default=_HERE / "configs")
    args = parser.parse_args()

    for code in ("hi",):
        result = balance(code, args.vocab_dir, args.config_dir,
                         args.root.resolve())
        before, after = result["before"], result["after"]
        print(f"[{code}] train {result['sizes']['train']}, "
              f"test {result['sizes']['test']}, overlap {result['overlap']}")
        print(f"      before: train {before['train']:.3f}  "
              f"test {before['test']:.3f}  gap {before['gap']:+.3f}")
        print(f"      after:  train {after['train']:.3f}  "
              f"test {after['test']:.3f}  gap {after['gap']:+.3f}")


if __name__ == "__main__":
    main()
