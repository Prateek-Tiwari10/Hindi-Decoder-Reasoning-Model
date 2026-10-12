"""Correctness and leakage checks for the Phase 3 reasoning dataset (§9, §10).

WHY THE SOLVER HERE IS WRITTEN TWICE OVER
-----------------------------------------
:func:`solve_independently` deliberately shares NO code with ``generator.py``.
It reads only what a reader of the dataset file can read -- the surface
``premises`` list and the ``query`` block -- and re-derives the total order with
Kahn's algorithm rather than Floyd-Warshall, then re-derives the answer from
that order.  If the generator ever mislabels a problem, the two paths disagree
and G2 fails.  A checker that imported the generator's own reasoning would
agree with it about everything, including its bugs.

The checks are pure functions over an already-generated record list.  Nothing
here generates data and nothing here writes files; ``run.py`` orchestrates and
``report.py`` renders.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Sequence

#: Marker for a check that ran and passed.
PASS = "PASS"
#: Marker for a check that ran and failed.
FAIL = "FAIL"


def check_result(name: str, passed: bool, detail: str,
            numbers: dict | None = None) -> dict:
    """Package one check outcome uniformly.

    Args:
        name: Check id, e.g. ``"G1"``.
        passed: Whether it passed.
        detail: One-line human-readable summary.
        numbers: Reported measurements for RESULTS_part1.md.

    Returns:
        A check-result dict.
    """
    return {"check": name, "status": PASS if passed else FAIL,
            "detail": detail, "numbers": numbers or {}}


# ---------------------------------------------------------------------------
# The independent solver (G2)
# ---------------------------------------------------------------------------


def topological_order(entities: Sequence[str],
                      edges: Sequence[tuple[str, str]]) -> list[str] | None:
    """Kahn's algorithm, refusing any input that admits more than one order.

    At every step exactly one entity may have in-degree zero.  If two do, the
    premises leave their relative position open and the problem has more than
    one valid answer; if none do, there is a cycle.  Both return None, which is
    what makes this function a uniqueness test as well as a sort.

    Args:
        entities: All entity names in the problem.
        edges: ``(greater, lesser)`` pairs.

    Returns:
        The unique total order greatest-first, or None if it is not unique.
    """
    successors: dict[str, list[str]] = {e: [] for e in entities}
    indegree: dict[str, int] = {e: 0 for e in entities}
    for greater, lesser in edges:
        successors[greater].append(lesser)
        indegree[lesser] += 1

    order: list[str] = []
    remaining = dict(indegree)
    for _ in range(len(entities)):
        ready = [e for e, d in remaining.items() if d == 0]
        if len(ready) != 1:
            return None
        current = ready[0]
        order.append(current)
        del remaining[current]
        for successor in successors[current]:
            if successor in remaining:
                remaining[successor] -= 1
    return order


def solve_independently(record: dict, config: dict) -> dict:
    """Re-derive a record's order and answer from its stated premises alone.

    Args:
        record: One dataset record.
        config: That language's config, for the yes/no labels.

    Returns:
        ``{"order": [...] | None, "answer": str | None, "reason": str}``.
        ``reason`` is non-empty only when the record could not be solved.
    """
    entities: list[str] = []
    for premise in record["premises"]:
        for side in ("a", "b"):
            if premise[side] not in entities:
                entities.append(premise[side])

    edges = [(p["a"], p["b"]) if p["rel"] == ">" else (p["b"], p["a"])
             for p in record["premises"]]
    order = topological_order(entities, edges)
    if order is None:
        return {"order": None, "answer": None,
                "reason": "premises do not determine a unique total order"}

    labels = config["labels"]
    query = record["query"]
    kind = query["kind"]

    if kind == "extremum":
        answer = order[0] if query["end"] == "max" else order[-1]
    elif kind == "ordering":
        answer = " > ".join(order)
    elif kind == "pairwise":
        a_greater = order.index(query["a"]) < order.index(query["b"])
        affirmative = a_greater if query["direction"] == "more" else not a_greater
        answer = labels["affirmative" if affirmative else "negative"]
    elif kind == "bound":
        above = order.index(query["entity"]) < order.index(query["anchor"])
        affirmative = above if query["direction"] == "more" else not above
        answer = labels["affirmative" if affirmative else "negative"]
    else:
        return {"order": order, "answer": None,
                "reason": f"unknown query kind {kind!r}"}

    return {"order": order, "answer": answer, "reason": ""}


# ---------------------------------------------------------------------------
# G1-G4, G8 -- per-record structural checks
# ---------------------------------------------------------------------------


def check_well_posed(records: Sequence[dict]) -> dict:
    """G1: every problem's premises pin down exactly one total order.

    Args:
        records: All records for one language.

    Returns:
        A check-result dict.
    """
    failures = [r["id"] for r in records
                if topological_order(
                    list(dict.fromkeys(
                        [p[s] for p in r["premises"] for s in ("a", "b")])),
                    [(p["a"], p["b"]) if p["rel"] == ">" else (p["b"], p["a"])
                     for p in r["premises"]]) is None]
    return check_result(
        "G1", not failures,
        f"{len(records) - len(failures)}/{len(records)} uniquely determined",
        {"total": len(records), "underdetermined": len(failures),
         "examples": failures[:5]})


def check_solver_agreement(records: Sequence[dict], config: dict) -> dict:
    """G2: the independent solver reproduces both ``answer`` and ``true_order``.

    Args:
        records: All records for one language.
        config: That language's config.

    Returns:
        A check-result dict.
    """
    answer_mismatches, order_mismatches = [], []
    for record in records:
        solved = solve_independently(record, config)
        if solved["answer"] != record["answer"]:
            answer_mismatches.append(record["id"])
        if solved["order"] != record["true_order"]:
            order_mismatches.append(record["id"])
    passed = not answer_mismatches and not order_mismatches
    return check_result(
        "G2", passed,
        f"{len(answer_mismatches)} answer / {len(order_mismatches)} order "
        f"mismatches over {len(records)} records",
        {"total": len(records), "answer_mismatches": len(answer_mismatches),
         "order_mismatches": len(order_mismatches),
         "examples": (answer_mismatches + order_mismatches)[:5]})


def check_non_adjacency(records: Sequence[dict]) -> dict:
    """G3: a relational pairwise question never queries a stated premise pair.

    Bound questions are excluded by construction: they compare an entity
    against a NUMBER, not against another entity, so "the queried pair" is not
    defined for them.  Their analogous constraint -- that the queried entity is
    not directly linked to the anchor -- is reported separately as the bound
    adjacency relaxation count.

    Args:
        records: All records for one language.

    Returns:
        A check-result dict.
    """
    checked, violations = 0, []
    for record in records:
        if record["question_type"] != "pairwise":
            continue
        if record["question_variant"] != "relation":
            continue
        checked += 1
        queried = frozenset((record["query"]["a"], record["query"]["b"]))
        stated = {frozenset((p["a"], p["b"])) for p in record["premises"]}
        if queried in stated:
            violations.append(record["id"])
    return check_result(
        "G3", not violations,
        f"{checked - len(violations)}/{checked} relational pairwise questions "
        f"query an underived pair",
        {"checked": checked, "violations": len(violations),
         "examples": violations[:5]})


def check_nonconsecutive_coverage(records: Sequence[dict],
                                  min_depth: int) -> dict:
    """G4: problems at depth >= min_depth carry a redundant non-consecutive premise.

    Two populations are reported, and the difference between them is the
    finding rather than a shortfall:

    * ALL records at depth >= min_depth;
    * the ELIGIBLE subset, excluding depth-3 relational pairwise items.

    A depth-3 problem has exactly one non-consecutive pair, ``(0, 2)``.  If the
    question queries it -- which §6.2 requires, since the queried pair must not
    be a premise -- then it cannot also be stated as a premise, so those items
    necessarily have zero non-consecutive premises.  The eligible fraction is
    the one that can meaningfully reach 100%.

    Args:
        records: All records for one language.
        min_depth: Depth from which non-consecutive premises are possible.

    Returns:
        A check-result dict.
    """
    deep = [r for r in records if r["depth"] >= min_depth]
    eligible = [r for r in deep
                if not (r["question_type"] == "pairwise"
                        and r["question_variant"] == "relation"
                        and r["depth"] == 3)]
    with_nc = sum(1 for r in deep if r["n_nonconsecutive_premises"] > 0)
    eligible_with_nc = sum(1 for r in eligible
                           if r["n_nonconsecutive_premises"] > 0)
    mean_all = (statistics.fmean([r["n_nonconsecutive_premises"] for r in deep])
                if deep else 0.0)
    mean_eligible = (statistics.fmean(
        [r["n_nonconsecutive_premises"] for r in eligible])
        if eligible else 0.0)
    passed = not eligible or eligible_with_nc == len(eligible)
    fraction = eligible_with_nc / len(eligible) if eligible else 1.0
    return check_result(
        "G4", passed,
        f"{fraction:.4f} of eligible depth>={min_depth} problems carry >=1 "
        f"non-consecutive premise, mean {mean_eligible:.2f} each; over ALL "
        f"depth>={min_depth} problems the mean is {mean_all:.2f}, below the "
        f"target of 1 because depth-3 relational pairwise items cannot carry "
        f"one",
        {"depth_ge_min": len(deep), "with_nonconsecutive": with_nc,
         "eligible": len(eligible), "eligible_with_nonconsecutive":
             eligible_with_nc, "eligible_fraction": round(fraction, 6),
         "mean_nonconsecutive_eligible": round(mean_eligible, 4),
         "mean_nonconsecutive_all": round(mean_all, 4),
         "structurally_excluded": len(deep) - len(eligible),
         "all_fraction": round(with_nc / len(deep), 6) if deep else 1.0})


def check_answer_extraction(records: Sequence[dict], config: dict,
                            extract) -> dict:
    """G8: the documented extraction rule recovers ``answer`` from ``completion``.

    Args:
        records: All records for one language.
        config: That language's config.
        extract: The extraction function under test
            (``generator.extract_answer``), passed in rather than imported so
            the rule being verified is the one Part 3 will call.

    Returns:
        A check-result dict.
    """
    failures = []
    for record in records:
        try:
            recovered = extract(config, record["completion"])
        except ValueError:
            failures.append(record["id"])
            continue
        if recovered != record["answer"]:
            failures.append(record["id"])
    return check_result(
        "G8", not failures,
        f"{len(records) - len(failures)}/{len(records)} answers recovered "
        f"by the rule 'text between the last answer cue and the final danda'",
        {"total": len(records), "failures": len(failures),
         "examples": failures[:5]})


# ---------------------------------------------------------------------------
# G5 -- sequence length, measured with the real Phase 2 tokenizer
# ---------------------------------------------------------------------------


def measure_lengths(records: Sequence[dict], tokenizer,
                    context_len: int = 512, warn_at: int = 400) -> dict:
    """G5: token lengths of ``prompt + completion``, per slice.

    Measured with the language's actual Phase 2 16K BPE tokenizer, loaded
    read-only.  Estimating from character counts would not do: Devanagari
    fertility is the whole reason bits-per-byte exists in this project, and a
    character-based estimate would be wrong by exactly the factor under study.

    Args:
        records: All records for one language.
        tokenizer: A loaded SentencePiece processor.
        context_len: The Phase 2 context length the data must fit inside.
        warn_at: Length above which a slice is flagged even though it fits.

    Returns:
        A check-result dict whose numbers carry per-slice mean/p99/max.
    """
    by_slice: dict[str, list[int]] = defaultdict(list)
    for record in records:
        text = record["prompt"] + record["completion"]
        by_slice[record["slice"]].append(len(tokenizer.encode(text)))

    per_slice, overflowing, warned = {}, [], []
    for tag, lengths in sorted(by_slice.items()):
        lengths.sort()
        p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
        summary = {"n": len(lengths),
                   "mean": round(statistics.fmean(lengths), 1),
                   "p99": p99, "max": lengths[-1]}
        per_slice[tag] = summary
        if summary["max"] >= context_len:
            overflowing.append(tag)
        elif summary["max"] > warn_at:
            warned.append(tag)

    overall_max = max(s["max"] for s in per_slice.values())
    return check_result(
        "G5", not overflowing,
        f"longest sequence {overall_max} tokens against a {context_len}-token "
        f"context ({context_len - overall_max} to spare)"
        + (f"; slices above {warn_at}: {warned}" if warned else ""),
        {"context_len": context_len, "warn_at": warn_at,
         "overall_max": overall_max, "overflowing": overflowing,
         "warned": warned, "per_slice": per_slice})


def measure_name_lengths(train_names: Sequence[str], test_names: Sequence[str],
                         tokenizer) -> dict:
    """§9.5: mean subword length of train vs test names.

    A gap here would be a confound rather than a leak: if test names were
    systematically longer in subwords than train names, a drop in test accuracy
    could be a tokenization effect rather than a failure to generalize over
    names.  It is the same fertility story as the Phase 2 BPB discussion.

    Args:
        train_names: The train pool.
        test_names: The test pool.
        tokenizer: A loaded SentencePiece processor.

    Returns:
        Mean, max and the train/test gap, in subword tokens.
    """
    def summarize(names: Sequence[str]) -> dict:
        lengths = [len(tokenizer.encode(n)) for n in names]
        return {"n": len(names), "mean": round(statistics.fmean(lengths), 3),
                "max": max(lengths), "min": min(lengths)}

    train, test = summarize(train_names), summarize(test_names)
    return {"train": train, "test": test,
            "gap_mean": round(test["mean"] - train["mean"], 3)}


# ---------------------------------------------------------------------------
# G6 -- leakage
# ---------------------------------------------------------------------------


def check_pool_disjointness(train_names: Sequence[str],
                            test_names: Sequence[str]) -> dict:
    """§9.1: the train and test name pools share no name.

    This is the primary leakage control.  Every test slice draws from the test
    pool only, so any name-to-answer association memorised during finetuning is
    worthless at test time.

    Args:
        train_names: The train pool.
        test_names: The test pool.

    Returns:
        A check-result dict.
    """
    overlap = sorted(set(train_names) & set(test_names))
    return check_result(
        "G6.names", not overlap,
        f"train pool {len(train_names)}, test pool {len(test_names)}, "
        f"{len(overlap)} shared",
        {"train_pool": len(train_names), "test_pool": len(test_names),
         "overlap": overlap})


def check_prompt_overlap(records: Sequence[dict]) -> dict:
    """§9.2: no prompt string appears in more than one split or slice.

    Args:
        records: All records for one language.

    Returns:
        A check-result dict naming the colliding slice pairs, if any.
    """
    owner: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for record in records:
        previous = owner.get(record["prompt"])
        if previous is not None:
            collisions.append((previous, record["slice"]))
        else:
            owner[record["prompt"]] = record["slice"]
    return check_result(
        "G6.prompts", not collisions,
        f"{len(records)} records, {len(owner)} distinct prompts, "
        f"{len(collisions)} collisions",
        {"records": len(records), "distinct_prompts": len(owner),
         "collisions": len(collisions),
         "colliding_pairs": sorted(set(collisions))[:5]})


def chi_square_p_value(statistic: float, dof: int) -> float:
    """Upper-tail p-value of a chi-square statistic, via Wilson-Hilferty.

    Implemented with the standard library rather than SciPy, which is not a
    dependency of this project.  The Wilson-Hilferty cube-root transform is
    accurate to about three decimal places for the degrees of freedom seen
    here (hundreds), which is far more precision than a pass/fail gate needs.

    Args:
        statistic: The chi-square statistic.
        dof: Degrees of freedom, at least 1.

    Returns:
        P(chi2_dof >= statistic).
    """
    if dof < 1:
        return 1.0
    if statistic <= 0:
        return 1.0
    ratio = (statistic / dof) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * dof)
    sigma = math.sqrt(2.0 / (9.0 * dof))
    return 1.0 - statistics.NormalDist().cdf((ratio - mean) / sigma)


def binomial_z(successes: int, trials: int, probability: float = 0.5) -> float:
    """Standardized deviation of a count from its binomial expectation.

    Args:
        successes: Observed count.
        trials: Number of trials.
        probability: Null success probability.

    Returns:
        The z-score, or 0.0 when there are no trials.
    """
    if trials <= 0:
        return 0.0
    expected = trials * probability
    sigma = math.sqrt(trials * probability * (1.0 - probability))
    return 0.0 if sigma == 0 else (successes - expected) / sigma


def check_label_balance(records: Sequence[dict], config: dict,
                        max_z: float = 3.5) -> dict:
    """§9.3: answer positions and yes/no labels match a fair coin.

    WHY THIS IS A Z-TEST AND NOT A FIXED TOLERANCE
    ----------------------------------------------
    Both distributions are fair coins BY CONSTRUCTION -- the question's
    comparison direction and the max/min choice are each drawn at p = 0.5 --
    so the null hypothesis is not merely plausible here, it is exactly true.
    What the check has to detect is a *generator bug* that breaks that, and the
    deviation such a bug produces does not shrink with n while sampling noise
    does (as 1/sqrt(n)).  A fixed absolute tolerance would therefore be too
    tight on small builds and too loose on large ones: at n = 300 a harmless
    0.03 deviation is one standard error, while at n = 40,000 the same 0.03 is
    twelve standard errors and is definitely a bug.  The raw deviation is still
    reported, because §9.3 asks for it.

    Ordering questions are excluded: their answer is the whole permutation, so
    "position of the answer" is undefined.

    Args:
        records: All records for one language.
        config: That language's config, for the yes/no labels.
        max_z: Standardized deviation above which the check fails.

    Returns:
        A check-result dict.
    """
    labels = config["labels"]
    boolean: Counter = Counter()
    extremum: Counter = Counter()
    for record in records:
        if record["query"]["kind"] in ("pairwise", "bound"):
            boolean[record["answer"]] += 1
        elif record["query"]["kind"] == "extremum":
            extremum[record["query"]["end"]] += 1

    def summarize(counter: Counter, first: str) -> dict:
        total = sum(counter.values())
        z = binomial_z(counter[first], total)
        deviation = abs(counter[first] / total - 0.5) if total else 0.0
        return {"counts": dict(counter), "n": total,
                "deviation": round(deviation, 6), "z": round(z, 3)}

    boolean_summary = summarize(boolean, labels["affirmative"])
    extremum_summary = summarize(extremum, "max")
    worst_z = max(abs(boolean_summary["z"]), abs(extremum_summary["z"]))

    return check_result(
        "G6.balance", worst_z <= max_z,
        f"yes/no {boolean_summary['deviation']:.4f} (z={boolean_summary['z']:+.2f}), "
        f"max/min {extremum_summary['deviation']:.4f} "
        f"(z={extremum_summary['z']:+.2f}); |z| limit {max_z}",
        {"max_z": max_z, "worst_z": round(worst_z, 3),
         "boolean": boolean_summary, "extremum": extremum_summary})


def check_name_answer_independence(records: Sequence[dict],
                                   min_p_value: float = 0.001,
                                   family_alpha: float = 0.01) -> dict:
    """§9.4: knowing a name tells you nothing about whether it is the answer.

    THE EXACT NULL, RATHER THAN A POOLED BASE RATE
    ----------------------------------------------
    For an extremum question of depth d, the answer is position 0 or position
    d-1, each with probability 1/2, and a uniformly sampled name lands in any
    position with probability 1/d.  So a name that appears in a depth-d problem
    is the answer with probability exactly ``1/d``, whatever d is.  Each name's
    expectation is therefore the sum of ``1/depth`` over its own appearances,
    not the pooled rate across the corpus -- which matters because the slices
    mix depths 2 through 6, and a pooled rate would charge a name that happened
    to appear in deep problems with an inflated expectation.

    Two instruments, both reported:

    * a **chi-square** goodness-of-fit statistic over all names (§9.4 names
      this explicitly), which asks whether the deviations as a whole are larger
      than sampling noise;
    * the **maximum standardized deviation** with a Bonferroni-corrected
      threshold, which asks whether any single name is an outlier once you
      account for having tested hundreds of them.

    The raw maximum deviation and the L1 summary are reported too, since §9.4
    asks for them, but they are not what the pass/fail decision rests on: a
    name seen five times that happens to be the answer twice shows a raw
    deviation of 0.07 and means nothing at all.

    Args:
        records: All records for one language.
        min_p_value: Chi-square p-value below which the check fails.
        family_alpha: Family-wise error rate for the per-name outlier test.

    Returns:
        A check-result dict.
    """
    appearances: Counter = Counter()
    answers: Counter = Counter()
    expected: dict[str, float] = defaultdict(float)
    variance: dict[str, float] = defaultdict(float)

    for record in records:
        if record["query"]["kind"] != "extremum":
            continue
        probability = 1.0 / record["depth"]
        for name in record["true_order"]:
            appearances[name] += 1
            expected[name] += probability
            variance[name] += probability * (1.0 - probability)
        answers[record["answer"]] += 1

    if not appearances:
        return check_result("G6.names_carry_no_label", True,
                            "no extremum questions to score", {})

    z_scores: dict[str, float] = {}
    raw_deviations: dict[str, float] = {}
    for name, count in appearances.items():
        sigma = math.sqrt(variance[name]) if variance[name] > 0 else 0.0
        observed = answers.get(name, 0)
        z_scores[name] = ((observed - expected[name]) / sigma) if sigma else 0.0
        raw_deviations[name] = observed / count - expected[name] / count

    chi_square = sum(z * z for z in z_scores.values())
    dof = max(1, len(z_scores) - 1)
    p_value = chi_square_p_value(chi_square, dof)

    ranked = sorted(z_scores.items(), key=lambda kv: -abs(kv[1]))
    max_abs_z = abs(ranked[0][1])
    z_threshold = statistics.NormalDist().inv_cdf(
        1.0 - family_alpha / (2.0 * len(z_scores)))

    max_raw = max(abs(d) for d in raw_deviations.values())
    l1 = statistics.fmean(abs(d) for d in raw_deviations.values())
    pooled_rate = sum(answers.values()) / sum(appearances.values())

    passed = p_value >= min_p_value and max_abs_z <= z_threshold
    return check_result(
        "G6.names_carry_no_label", passed,
        f"chi2={chi_square:.1f} on {dof} df, p={p_value:.3f}; max |z|="
        f"{max_abs_z:.2f} against a Bonferroni limit of {z_threshold:.2f}; "
        f"max raw deviation {max_raw:.4f}, mean |deviation| {l1:.4f}",
        {"names_scored": len(z_scores), "pooled_answer_rate":
             round(pooled_rate, 6), "chi_square": round(chi_square, 3),
         "dof": dof, "p_value": round(p_value, 6),
         "min_p_value": min_p_value, "max_abs_z": round(max_abs_z, 3),
         "z_threshold": round(z_threshold, 3), "family_alpha": family_alpha,
         "max_raw_deviation": round(max_raw, 6),
         "mean_abs_deviation": round(l1, 6),
         "worst": [{"name": n, "z": round(z, 3),
                    "raw_deviation": round(raw_deviations[n], 4),
                    "appearances": appearances[n],
                    "expected": round(expected[n], 2),
                    "observed": answers.get(n, 0)}
                   for n, z in ranked[:5]]})


def check_name_length_gap(train_names: Sequence[str],
                          test_names: Sequence[str], tokenizer,
                          tolerance: float = 0.15) -> dict:
    """§9.5 as a gate: train and test names cost the same in subwords.

    Reported as a check rather than only as a number because it is a confound
    with teeth.  Test accuracy is meant to fall only because the names are
    unseen; if the test names were also systematically longer in subwords, a
    fall could not be attributed to either cause.  The pools are stratified by
    subword length (``balance_pools.py``) so this gate holds by construction,
    and it is asserted here so that editing a pool by hand cannot silently
    reintroduce the imbalance.

    Args:
        train_names: The train pool.
        test_names: The test pool.
        tokenizer: A loaded SentencePiece processor.
        tolerance: Largest acceptable difference in mean tokens per name.

    Returns:
        A check-result dict.
    """
    train_mean = statistics.fmean(len(tokenizer.encode(n)) for n in train_names)
    test_mean = statistics.fmean(len(tokenizer.encode(n)) for n in test_names)
    gap = test_mean - train_mean
    return check_result(
        "G6.name_length", abs(gap) <= tolerance,
        f"train {train_mean:.3f} vs test {test_mean:.3f} tokens per name, "
        f"gap {gap:+.3f} (tolerance {tolerance})",
        {"train_mean": round(train_mean, 4), "test_mean": round(test_mean, 4),
         "gap": round(gap, 4), "tolerance": tolerance})


def question_sentence(record: dict) -> str:
    """The question sentence alone, split off from the premises.

    A prompt is ``<cue> <premise.> <premise.> <question>\n<answer cue>``.  Every
    premise ends in a danda and the question does not contain one, so the text
    after the last danda on the first line is exactly the question.

    Args:
        record: One dataset record.

    Returns:
        The question sentence, stripped.
    """
    body = record["prompt"].split("\n")[0]
    return body.rsplit("\u0964", 1)[-1].strip()


def check_extremum_gender_neutral(records: Sequence[dict], config: dict) -> dict:
    """Gender agreement must not leak the answer's gender into the prompt.

    Turning on predicate agreement (revision Fix B) creates a leak that is easy
    to introduce and invisible once the data is written: if the extremum
    question agreed with the answer -- "सबसे लम्बी कौन है?" -- it would announce
    that the answer is feminine and roughly halve the candidate set for free.

    The question therefore takes the masculine form whatever the answer's
    gender, matching the grammatically masculine interrogative (कौन / को).  The
    final sentence inside the COMPLETION does agree, which is harmless: the
    completion is the target, not the input.

    The assertion is made on the QUESTION SENTENCE ALONE, not on the whole
    prompt.  The premises in the same prompt carry agreeing adjectives of their
    own and will legitimately contain feminine forms whenever a premise has a
    feminine subject -- so testing the whole prompt would fail on correct data.
    What must be gender-neutral is only the question.

    Args:
        records: All records for one language.
        config: That language's config, for the adjective tables.

    Returns:
        A check-result dict.
    """
    checked, violations = 0, []
    for record in records:
        if record["query"]["kind"] != "extremum":
            continue
        domain = config["domains"][record["domain"]]
        direction = "more" if record["query"]["end"] == "max" else "less"
        forms = domain[direction]
        if forms["m"] == forms["f"]:
            continue                      # invariant adjective: nothing to leak
        checked += 1
        question = question_sentence(record)
        if forms["m"] not in question or forms["f"] in question:
            violations.append(record["id"])
    return check_result(
        "G6.extremum_gender", not violations,
        f"{checked - len(violations)}/{checked} extremum prompts use the "
        f"gender-neutral (masculine) adjective, leaking nothing about the "
        f"answer's gender",
        {"checked": checked, "violations": len(violations),
         "examples": violations[:5]})
