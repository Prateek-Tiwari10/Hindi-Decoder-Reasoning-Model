"""Build the large name pools for the performance build.

WHY THIS EXISTS
---------------
Part 3A established that the models were not copying entity names out of the
prompt; they were recalling them from the 220-name pool they were finetuned
on.  With 40,000 training problems at depth 2-3 each name appeared roughly 455
times, which is ample for a 25.9M-parameter model to memorise the inventory and
never learn the copy operation the task actually requires.

The lever is repetitions per name.  Driving that down forces the model to read
names out of its context instead of recalling them, because no individual name
is seen often enough to be worth storing.  This module builds the larger pool.

HOW THE NAMES ARE MADE
----------------------
Three sources, in descending order of confidence:

1. the authored base pools from Part 1 -- real, high-frequency given names;
2. ``base + suffix`` compounds, which is the dominant productive pattern in
   Hindi: सुरेशकुमार, रामप्रसाद, हरिबहादुर, गीतादेवी, सीताकुमारी.
   These are not invented forms -- name plus an honorific or theophoric second
   element is how a very large share of real Hindi given names are
   built;
3. ``element + element`` compounds over a curated list of short theophoric and
   virtue morphemes (रामकृष्ण, शिवशंकर, हरिनारायण), restricted so the second
   element is one that genuinely heads a name.

Every candidate passes a tokenization check against that language's frozen 16K
tokenizer before it is accepted, so the pool cannot be inflated with names that
shatter into subword fragments.

WHAT IS FLAGGED RATHER THAN HIDDEN
----------------------------------
Composition is mechanical and will produce some combinations a native speaker
would not choose.  The longest and most heavily fragmented survivors are
written to the report for human review rather than presented as certainly
natural.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent

#: Second elements appended to a male given name.  Each is a real, productive
#: name-final morpheme in the language concerned.
MALE_SUFFIXES = {
    "hi": ["कुमार", "प्रसाद", "चंद्र", "नाथ", "लाल", "देव", "राज", "किशोर",
           "मोहन", "प्रकाश", "शंकर", "पाल", "दत्त", "नारायण", "स्वरूप",
           "चरण", "सिंह", "बाबू", "दास", "रतन"],
}

#: Second elements appended to a female given name.  Female compounding is less
#: productive than male, so these lists are deliberately shorter.
FEMALE_SUFFIXES = {
    "hi": ["देवी", "कुमारी", "रानी", "लता", "माला", "बाई"],
}

#: Short morphemes that open a compound male name.
MODIFIERS = {
    "hi": ["राम", "शिव", "हरि", "कृष्ण", "गोपाल", "गोविंद", "माधव", "केशव",
           "मुकुंद", "नारायण", "देव", "सूर्य", "चंद्र", "भानु", "अमर", "प्रेम",
           "धर्म", "सत्य", "ज्ञान", "जय", "विजय", "वीर", "बाल", "नंद", "रघु",
           "लक्ष्मण", "भरत", "शंकर", "महेश", "गणेश", "विश्व", "जग", "लोक",
           "मंगल", "शुभ", "सुख", "रत्न", "कमल", "पुष्प", "इंद्र"],
}

#: Morphemes that can HEAD a compound male name (stand as its second element).
HEADS = {
    "hi": ["राम", "कृष्ण", "शंकर", "नारायण", "देव", "चंद्र", "प्रसाद", "कुमार",
           "नाथ", "लाल", "मोहन", "गोपाल", "किशोर", "प्रकाश", "दत्त", "पाल",
           "सिंह", "राज", "वीर", "स्वरूप", "चरण", "दास", "रतन", "बहादुर"],
}

#: A name that costs more subword tokens than this is rejected.  The point of
#: the pool is to teach copying; a name that shatters into many fragments makes
#: copying harder for reasons that have nothing to do with reasoning, and would
#: also inflate sequence length.
MAX_NAME_TOKENS = 7


def load_pool(path: Path) -> list[tuple[str, str]]:
    """Read a pool file as ``(name, gender)`` pairs.

    Args:
        path: Pool file.

    Returns:
        The entries in file order.
    """
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            entries.append((parts[0], parts[1] if len(parts) > 1 else "m"))
    return entries


def is_simple(name: str, suffixes: list[str]) -> bool:
    """True if a name is not already a compound of one of these suffixes.

    Stops ``रामकुमार`` from becoming ``रामकुमारकुमार`` and keeps the
    compounds one morpheme deep, which is where real names sit.

    Args:
        name: Candidate base.
        suffixes: The suffix inventory.

    Returns:
        Whether the name may take a suffix.
    """
    return not any(name.endswith(suffix) for suffix in suffixes)


def compose(code: str, base: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Generate every candidate name for one language.

    Args:
        code: ``"hi"``.
        base: The authored base pool.

    Returns:
        ``(name, gender)`` candidates, base names first.
    """
    male_suffixes = MALE_SUFFIXES[code]
    female_suffixes = FEMALE_SUFFIXES[code]
    seen = {name for name, _ in base}
    out = list(base)

    for name, gender in base:
        suffixes = male_suffixes if gender == "m" else female_suffixes
        if not is_simple(name, suffixes):
            continue
        for suffix in suffixes:
            candidate = name + suffix
            if candidate not in seen:
                seen.add(candidate)
                out.append((candidate, gender))

    for modifier in MODIFIERS[code]:
        for head in HEADS[code]:
            if modifier == head:
                continue
            candidate = modifier + head
            if candidate not in seen:
                seen.add(candidate)
                out.append((candidate, "m"))
    return out


def screen(candidates: list[tuple[str, str]], tokenizer
           ) -> tuple[list[tuple[str, str]], dict]:
    """Reject candidates the tokenizer handles badly.

    Args:
        candidates: ``(name, gender)`` pairs.
        tokenizer: The language's frozen 16K SentencePiece model.

    Returns:
        ``(accepted, rejection_counts)``.
    """
    unk = tokenizer.unk_id()
    accepted, rejected = [], {"too_long": 0, "unk": 0, "roundtrip": 0}
    for name, gender in candidates:
        ids = tokenizer.encode(name, out_type=int)
        if any(i == unk for i in ids):
            rejected["unk"] += 1
            continue
        if tokenizer.decode(ids) != name:
            rejected["roundtrip"] += 1
            continue
        if len(ids) > MAX_NAME_TOKENS:
            rejected["too_long"] += 1
            continue
        accepted.append((name, gender))
    return accepted, rejected


def apportion(bucket_sizes: dict[int, int], total_test: int) -> dict[int, int]:
    """Largest-remainder apportionment of the test quota across buckets.

    Args:
        bucket_sizes: Bucket key to size.
        total_test: Names that must land in the test pool.

    Returns:
        Bucket key to its integer quota.
    """
    total = sum(bucket_sizes.values())
    exact = {k: n * total_test / total for k, n in bucket_sizes.items()}
    quota = {k: int(v) for k, v in exact.items()}
    remaining = total_test - sum(quota.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))
    for key in order[:remaining]:
        quota[key] += 1
    return quota


def split_stratified(accepted: list[tuple[str, str]], tokenizer,
                     test_fraction: float, seed: int
                     ) -> tuple[list, list, dict]:
    """Split into train and test pools, matched on subword length.

    The fertility control from Part 1 is preserved exactly: names are bucketed
    by their subword length and each bucket is split in the same proportion, so
    the two pools have the same mean tokens per name.  Without it a drop in
    test accuracy could be a tokenization effect rather than a failure to
    generalize over names.

    Args:
        accepted: Screened candidates.
        tokenizer: The language's tokenizer.
        test_fraction: Share going to the test pool.
        seed: Shuffle seed.

    Returns:
        ``(train, test, stats)``.
    """
    import random
    from collections import defaultdict

    lengths = {name: len(tokenizer.encode(name)) for name, _ in accepted}
    buckets: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for entry in accepted:
        buckets[lengths[entry[0]]].append(entry)

    n_test = round(test_fraction * len(accepted))
    quota = apportion({k: len(v) for k, v in buckets.items()}, n_test)
    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(buckets):
        members = sorted(buckets[key])
        rng.shuffle(members)
        test += members[:quota[key]]
        train += members[quota[key]:]

    train.sort(key=lambda e: (e[1], e[0]))
    test.sort(key=lambda e: (e[1], e[0]))
    stats = {
        "train_mean_tokens": round(
            statistics.fmean(lengths[n] for n, _ in train), 4),
        "test_mean_tokens": round(
            statistics.fmean(lengths[n] for n, _ in test), 4),
    }
    stats["gap"] = round(stats["test_mean_tokens"]
                         - stats["train_mean_tokens"], 4)
    return train, test, stats


HEADER = """# {language} given-name pool -- {side} side, PERFORMANCE BUILD.
#
# Format: <name>\\t<gender>   where gender is `m` or `f`.
#
# BUILT BY `python -m data.reasoning.expand_names`.  Part 3A showed the models
# were recalling names from the 220-name Part 1 pool rather than copying them
# out of the prompt -- at ~455 repetitions per name during finetuning, that is
# the cheaper strategy.  This pool is large enough that memorising it is not.
#
# Sources: the authored Part 1 base names; `base + suffix` compounds, the
# dominant productive pattern in Hindi; and `element + element`
# compounds over short theophoric morphemes.  Every name is screened against
# this language's frozen 16K tokenizer.
#
# The split is STRATIFIED BY SUBWORD LENGTH, so the train and test pools are
# matched on tokens per name -- the Part 1 fertility control, preserved.
# The two pools are DISJOINT.
#
# HUMAN REVIEW REQUESTED: composition is mechanical.  See RESULTS_perf.md for
# the longest and most fragmented survivors, which are the ones most likely to
# read as artificial.
"""


def build(code: str, language: str, root: Path, test_fraction: float,
          seed: int) -> dict:
    """Build and write one language's performance pools.

    Args:
        code: ``"hi"``.
        language: Language name, for the tokenizer path.
        root: Repository root.
        test_fraction: Share of names going to the test pool.
        seed: Split seed.

    Returns:
        A summary of what was built.
    """
    import sentencepiece as spm

    vocab = _HERE / "vocab"
    base = (load_pool(vocab / f"{code}_names_train.txt")
            + load_pool(vocab / f"{code}_names_test.txt"))

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(root / "tokenizers" / language / "tokenizer"
                       / f"{language}_bpe_16000.model"))

    candidates = compose(code, base)
    accepted, rejected = screen(candidates, tokenizer)
    train, test, stats = split_stratified(accepted, tokenizer, test_fraction,
                                          seed)

    for side, entries in (("TRAIN", train), ("TEST", test)):
        path = vocab / f"{code}_names_{side.lower()}_perf.txt"
        lines = [HEADER.format(language=language.title(), side=side), ""]
        lines += [f"{name}\t{gender}" for name, gender in entries]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lengths = sorted(((len(tokenizer.encode(n)), n) for n, _ in accepted),
                     reverse=True)
    return {
        "code": code, "language": language,
        "base_pool": len(base), "candidates": len(candidates),
        "accepted": len(accepted), "rejected": rejected,
        "train_pool": len(train), "test_pool": len(test),
        **stats,
        "longest_names_for_review": [{"name": n, "tokens": t}
                                     for t, n in lengths[:12]],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=_HERE.parent.parent)
    parser.add_argument("--test-fraction", type=float, default=0.24,
                        help="matches the Part 1 220:70 ratio")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    summaries = [build(code, language, root, args.test_fraction, args.seed)
                 for code, language in (("hi", "hindi"),)]

    report = (args.report
              or root / "reports" / "gpt_finetune" / "finetune_data" / "perf_name_pools.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"languages": summaries}, ensure_ascii=False,
                                 indent=2), encoding="utf-8")

    print()
    for s in summaries:
        print(f"[{s['code']}] base {s['base_pool']} -> candidates "
              f"{s['candidates']} -> accepted {s['accepted']} "
              f"(rejected {s['rejected']})")
        print(f"      train {s['train_pool']}, test {s['test_pool']}, "
              f"token-length gap {s['gap']:+.4f} "
              f"(train {s['train_mean_tokens']:.3f}, "
              f"test {s['test_mean_tokens']:.3f})")
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main()
