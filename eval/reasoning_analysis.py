"""Focused analyses over the scored generations.

Computation only -- every function takes score rows and returns numbers.
Plotting lives in ``figures.py`` and formatting in ``report.py``.

The centrepiece is :func:`bound_bug` (§4).  Part 2 surfaced a single example in
which a model assembled the correct order and then contradicted it at the
bound-propagation step, answering in the direction the *question* proposed.
That is either an anecdote or a mechanism, and the difference is measurable:
a mechanism shows up as an elevated self-contradiction rate confined to bound
questions, a polarity-echo rate far above the 50% chance baseline, and the same
pattern in both languages.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from eval import reasoning_metrics as eval_metrics


def load_rows(path: Path) -> list[dict]:
    """Read a score-row file.

    Args:
        path: ``rows_<language>_<tag>.jsonl``.

    Returns:
        The rows.
    """
    return [json.loads(line) for line
            in Path(path).read_text(encoding="utf-8").splitlines()]


def wilson_interval(successes: int, total: int,
                    z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because several rates here sit at
    or near zero, where the normal interval is nonsense (it extends below 0 and
    has zero width at exactly 0).

    Args:
        successes: Count of successes.
        total: Number of trials.
        z: Normal quantile; 1.96 for 95%.

    Returns:
        ``(low, high)``.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / total
                            + z * z / (4 * total * total)) / denominator)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bound_bug(rows: Sequence[dict]) -> dict:
    """§4: characterize the numeric-bound failure.

    Args:
        rows: Score rows for one language and checkpoint.

    Returns:
        Per-depth statistics for bound questions, the relational-pairwise
        contrast, and the polarity-echo measurement.
    """
    bound = [r for r in rows if r["query_kind"] == "bound"]
    relational = [r for r in rows
                  if r["question_type"] == "pairwise"
                  and r["question_variant"] == "relation"]

    def gap_stats(subset: Sequence[dict]) -> dict:
        """Chain-correct, EM, and the gap between them."""
        n = len(subset)
        if not n:
            return {"n": 0}
        chain_ok = [r for r in subset if r["chain_correct"]]
        gap = [r for r in chain_ok if not r["em"]]
        low, high = wilson_interval(len(gap), n)
        return {
            "n": n,
            "chain_correct": len(chain_ok) / n,
            "em": sum(1 for r in subset if r["em"]) / n,
            "gap_count": len(gap),
            "gap_rate": len(gap) / n,
            "gap_ci95": [low, high],
            # Conditional form: given the model reasoned correctly, how often
            # did it still get the answer wrong?  This is the cleaner statement
            # of the bug, because it removes variation in chain accuracy.
            "gap_given_chain_correct": (len(gap) / len(chain_ok)
                                        if chain_ok else 0.0),
            "chain_correct_n": len(chain_ok),
        }

    by_depth = {str(depth): gap_stats([r for r in bound if r["depth"] == depth])
                for depth in sorted({r["depth"] for r in bound})}

    # Polarity echo, measured on bound ERRORS only.  A model echoing the
    # question's polarity answers affirmatively whatever its chain says, so the
    # echo rate is the affirmative rate among errors; 50% is chance.
    errors = [r for r in bound if not r["em"]]
    echoes = sum(1 for r in errors if r.get("polarity_echo"))
    echo_low, echo_high = wilson_interval(echoes, len(errors))

    # And among the subset where the bug is defined: chain right, answer wrong.
    self_contradictions = [r for r in bound
                           if r["chain_correct"] and not r["em"]]
    sc_echoes = sum(1 for r in self_contradictions if r.get("polarity_echo"))

    return {
        "bound": gap_stats(bound),
        "bound_by_depth": by_depth,
        "relational_pairwise": gap_stats(relational),
        "polarity_echo": {
            "errors": len(errors),
            "echoes": echoes,
            "rate": echoes / len(errors) if errors else 0.0,
            "ci95": [echo_low, echo_high],
            "chance": 0.5,
        },
        "polarity_echo_on_self_contradictions": {
            "n": len(self_contradictions),
            "echoes": sc_echoes,
            "rate": (sc_echoes / len(self_contradictions)
                     if self_contradictions else 0.0),
        },
        "bound_step_disagrees_with_chain": sum(
            1 for r in bound if r.get("bound_step_matches_chain") is False),
    }


def error_taxonomy(rows: Sequence[dict]) -> dict:
    """§6.8: bucket every error into one of four kinds.

    The buckets are ordered and mutually exclusive, most specific first, so
    every error lands in exactly one and the counts sum.

    Args:
        rows: Score rows.

    Returns:
        Counts and rates per bucket, with example ids.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["em"]:
            continue
        if not row["format_valid"]:
            bucket = "format invalid"
        elif row["chain_correct"] and row["answer_follows_own_chain"] is False:
            bucket = "bound self-contradiction" \
                if row["query_kind"] == "bound" else "self-contradiction (other)"
        elif not row["chain_correct"]:
            bucket = "chain wrong"
        else:
            bucket = "other"
        buckets[bucket].append(row["id"])

    total_errors = sum(len(v) for v in buckets.values())
    return {
        "total_scored": len(rows),
        "total_errors": total_errors,
        "buckets": {name: {"count": len(ids),
                           "share_of_errors": (len(ids) / total_errors
                                               if total_errors else 0.0),
                           "examples": ids[:3]}
                    for name, ids in sorted(buckets.items(),
                                            key=lambda kv: -len(kv[1]))},
    }


def depth_curve(rows: Sequence[dict], slices: Sequence[str]) -> dict:
    """EM and chain accuracy per slice, for the depth-generalization panel.

    Args:
        rows: Score rows.
        slices: Slice tags to include, in order.

    Returns:
        Slice tag to its summary.
    """
    by_slice = eval_metrics.group_by(rows, "slice")
    return {tag: eval_metrics.summarize(by_slice.get((tag,), []))
            for tag in slices}


def ablation(primary: Sequence[dict], relonly: Sequence[dict],
             slices: Sequence[str]) -> dict:
    """§2.7: mixture-trained versus relation-only, on relation slices only.

    Restricted to slices the relation-only model was actually trained for, so
    the comparison is like-for-like.  Scoring it on numeric formats it never
    saw would produce a number with no interpretation.

    Args:
        primary: Score rows from the mixture-trained checkpoint.
        relonly: Score rows from the relation-only checkpoint.
        slices: Relation slice tags.

    Returns:
        Per-slice EM for both, the difference, and a verdict.
    """
    primary_by = eval_metrics.group_by(primary, "slice")
    relonly_by = eval_metrics.group_by(relonly, "slice")

    per_slice = {}
    for tag in slices:
        a = eval_metrics.summarize(primary_by.get((tag,), []))
        b = eval_metrics.summarize(relonly_by.get((tag,), []))
        if not a.get("n") or not b.get("n"):
            continue
        difference = a["em"] - b["em"]
        # Two independent proportions; the standard error of their difference.
        se = math.sqrt(a["em"] * (1 - a["em"]) / a["n"]
                       + b["em"] * (1 - b["em"]) / b["n"])
        per_slice[tag] = {
            "primary_em": a["em"], "relonly_em": b["em"],
            "difference": difference, "standard_error": se,
            "z": difference / se if se else 0.0,
            "significant": abs(difference) > 1.96 * se if se else False,
            "n_primary": a["n"], "n_relonly": b["n"],
        }

    # Multiple comparisons: five slices are tested per language, so at the
    # uncorrected 5% level roughly one spurious result per language is
    # expected.  A Bonferroni-corrected threshold is applied before anything is
    # called a difference -- the same discipline the Part 2 learning-rate sweep
    # used, and for the same reason.
    from statistics import NormalDist

    comparisons = max(1, len(per_slice))
    corrected_z = NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons))
    for value in per_slice.values():
        value["significant_corrected"] = abs(value["z"]) > corrected_z
        value["corrected_threshold"] = corrected_z

    significant = [t for t, v in per_slice.items()
                   if v["significant_corrected"]]
    raw_only = [t for t, v in per_slice.items()
                if v["significant"] and not v["significant_corrected"]]
    helps = [t for t in significant if per_slice[t]["difference"] > 0]
    hurts = [t for t in significant if per_slice[t]["difference"] < 0]

    if not significant:
        verdict = ("neutral -- no slice survives a Bonferroni-corrected test "
                   f"across {comparisons} comparisons"
                   + (f" (though {raw_only} would pass uncorrected)"
                      if raw_only else ""))
    elif helps and not hurts:
        verdict = (f"the mixture HELPS -- significantly better on {helps} "
                   f"after correction")
    elif hurts and not helps:
        verdict = (f"the mixture HURTS -- relation-only is significantly "
                   f"better on {hurts} after correction")
    else:
        verdict = (f"mixed -- mixture better on {helps}, relation-only better "
                   f"on {hurts}")

    return {"per_slice": per_slice, "verdict": verdict,
            "comparisons": comparisons,
            "corrected_threshold": corrected_z,
            "significant_uncorrected_only": raw_only}


def name_generalization(rows: Sequence[dict], val_em: float) -> dict:
    """Contrast in-distribution EM on held-out names against Part 2's val-EM.

    The two are matched on depth, task family and question mix; the systematic
    difference is that validation drew names from the TRAIN pool and every test
    slice draws from the disjoint TEST pool.  So the drop between them measures
    how much of the validation score was carried by the names themselves --
    which is exactly what Part 1's primary leakage control was built to expose.

    Args:
        rows: Score rows for the in-distribution relation slice.
        val_em: Part 2's validation exact match for this language.

    Returns:
        The two numbers and their difference.
    """
    summary = eval_metrics.summarize(rows)
    return {
        "val_em_train_names": val_em,
        "test_em_heldout_names": summary["em"],
        "drop": val_em - summary["em"],
        "chain_correct_heldout": summary["chain_correct"],
        "n": summary["n"],
    }


def load_name_pools(root: Path, language: str, suffix: str = ""
                    ) -> tuple[set[str], set[str]]:
    """Read the train and test name pools for one language.

    Args:
        root: Repository root.
        language: Language name.
        suffix: Pool variant, e.g. ``"_perf"`` for the performance build.
            Which pool counts as "train" is model-specific: it must be the
            pool the model under test was actually FINETUNED on, since the
            whole point of the measurement is whether emitted names come from
            that inventory.

    Returns:
        ``(train_names, test_names)``.
    """
    code = "hi"
    pools = []
    for side in ("train", "test"):
        path = (root / "data" / "reasoning" / "vocab"
                / f"{code}_names_{side}{suffix}.txt")
        names = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.add(line.split()[0])
        pools.append(names)
    return pools[0], pools[1]


def name_fidelity(rows: Sequence[dict], records: dict, train_pool: set[str],
                  test_pool: set[str], parser) -> dict:
    """Does the model COPY entity names from the prompt, or recall them?

    This is the mechanism behind the depth collapse, and it is measurable
    directly.  Every entity in a correct chain must be one of the entities the
    prompt named.  If instead the model emits names that appear nowhere in its
    input -- and especially if those names come from the pool it was FINETUNED
    on -- then it is not copying from context at all; it is retrieving from a
    memorised inventory, which cannot survive a held-out name pool.

    Three quantities are separated, because they carry different weight:

    * ``chain_length_matches``: did the model even emit a chain of the right
      length?  A model that always emits three entities has not generalized
      past the depths it saw, whatever the names are.
    * ``hallucinated_rate``: share of emitted entities absent from the prompt.
    * ``hallucinated_from_train_pool``: of those, how many are names the model
      saw during finetuning.  This is the decisive number: names it was
      trained on appearing in answers to prompts that never mention them is
      recall, not copying.

    Args:
        rows: Score rows carrying ``generated``.
        records: Record id to the dataset record.
        train_pool: Finetuning name pool.
        test_pool: Held-out name pool used by every test slice.
        parser: The completion parser for this language.

    Returns:
        Rates overall and per depth.
    """
    per_depth: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "length_ok": 0, "entities": 0, "hallucinated": 0,
                 "from_train": 0, "from_test": 0, "chains_all_in_prompt": 0})
    overall = {"n": 0, "entities": 0, "hallucinated": 0, "from_train": 0,
               "from_test": 0, "length_ok": 0, "chains_all_in_prompt": 0}
    examples: list[dict] = []

    for row in rows:
        # Keyed on (slice, id): Part 1 numbers records per slice, so the same
        # id appears in several slices and an id-only lookup silently pairs a
        # generation with a different slice's record.
        record = records.get((row["slice"], row["id"]))
        if record is None or "generated" not in row:
            continue
        chain = parser.parse(row["generated"], row["id"]).chain
        if not chain:
            continue
        prompt_names = set(record["true_order"])
        depth = record["depth"]
        bucket = per_depth[depth]

        length_ok = len(chain) == depth
        hallucinated = [e for e in chain if e not in prompt_names]
        from_train = [e for e in hallucinated if e in train_pool]
        from_test = [e for e in hallucinated
                     if e in test_pool and e not in prompt_names]

        for target in (bucket, overall):
            target["n"] += 1
            target["length_ok"] += length_ok
            target["entities"] += len(chain)
            target["hallucinated"] += len(hallucinated)
            target["from_train"] += len(from_train)
            target["from_test"] += len(from_test)
            target["chains_all_in_prompt"] += not hallucinated

        if hallucinated and from_train and len(examples) < 5:
            examples.append({
                "id": row["id"], "depth": depth,
                "true_order": record["true_order"],
                "emitted_chain": chain,
                "hallucinated": hallucinated,
                "from_train_pool": from_train,
                "generated": row["generated"][:300],
            })

    def finish(bucket: dict) -> dict:
        """Turn counts into rates."""
        n, entities = bucket["n"], bucket["entities"]
        return {
            "n": n,
            "chain_length_matches": bucket["length_ok"] / n if n else 0.0,
            "entities_emitted": entities,
            "hallucinated_rate": (bucket["hallucinated"] / entities
                                  if entities else 0.0),
            "hallucinated_from_train_pool": bucket["from_train"],
            "hallucinated_from_test_pool_elsewhere": bucket["from_test"],
            "share_of_hallucinations_from_train_pool":
                (bucket["from_train"] / bucket["hallucinated"]
                 if bucket["hallucinated"] else 0.0),
            "chains_entirely_from_prompt": (bucket["chains_all_in_prompt"] / n
                                            if n else 0.0),
        }

    return {
        "overall": finish(overall),
        "by_depth": {str(d): finish(per_depth[d]) for d in sorted(per_depth)},
        "examples": examples,
    }
