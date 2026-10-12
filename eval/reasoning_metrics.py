"""Scoring for one generated completion, and aggregation across slices.

THE CENTRAL DISTINCTION
-----------------------
Exact match collapses two very different failures into one number.  Every
score here keeps them apart by asking two questions of each generation:

* is the emitted ``क्रम:`` chain the true order?  (``chain_correct``)
* is the emitted final label entailed by the chain the model itself wrote,
  right or wrong?  (``answer_follows_own_chain``)

Crossing those gives the 2x2 of §2.4, and the off-diagonal cell "chain correct,
answer contradicts it" is the footprint of the numeric-bound bug.  A model that
reasons correctly and then misreads its own conclusion is a different object
from one that cannot order the entities at all, and only this split
distinguishes them.

Computation lives here; nothing in this module plots or prints.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

from eval.reasoning_parser import CompletionParser, ParsedCompletion


def entailed_label(chain: Sequence[str], query: dict, labels: dict
                   ) -> str | None:
    """The label the emitted chain entails, under the question that was asked.

    Deliberately evaluated against the MODEL'S chain rather than the true
    order: the question is whether the model's own conclusion follows from its
    own reasoning, which is what separates a reasoning error from a read-off
    error.

    Args:
        chain: Entities in the emitted chain, greatest first.
        query: The record's ``query`` block.
        labels: The language's affirmative/negative labels.

    Returns:
        The entailed label, or None if the chain does not contain the entities
        the question refers to (a degenerate generation).
    """
    kind = query["kind"]
    if not chain:
        return None

    if kind == "extremum":
        return chain[0] if query["end"] == "max" else chain[-1]

    if kind == "ordering":
        return " > ".join(chain)

    if kind == "pairwise":
        first, second = query["a"], query["b"]
        if first not in chain or second not in chain:
            return None
        greater = chain.index(first) < chain.index(second)
        affirmative = greater if query["direction"] == "more" else not greater
        return labels["affirmative" if affirmative else "negative"]

    if kind == "bound":
        entity, anchor = query["entity"], query["anchor"]
        if entity not in chain or anchor not in chain:
            return None
        above = chain.index(entity) < chain.index(anchor)
        affirmative = above if query["direction"] == "more" else not above
        return labels["affirmative" if affirmative else "negative"]

    return None


def copy_rate(record: dict, parsed: ParsedCompletion) -> dict:
    """§6: how much of the emitted chain was copied out of the prompt.

    WHY THIS METRIC EXISTS
    ----------------------
    Part 3A found the models recalling names from a small memorised inventory
    instead of copying them out of the prompt: at ~455 repetitions per name,
    memorising is the cheaper strategy.  The Perf build's larger name pools
    were the lever against that, and the scaling study has to show the fix
    HELD at every training-set size -- in particular at 10k, the smallest and
    lowest-recurrence subset, where a model sees each name only ~7 times.

    A name in the emitted chain that is not in the prompt is, by construction,
    invented: every entity a problem asks about is stated in its own prompt.
    So the fraction of chain names present in the prompt is a direct read on
    whether the model is copying or confabulating.

    Substring containment is the right test rather than token equality.  The
    prompt is Devanagari text with postpositions written attached
    (``रामको``, ``सुमनसँग``), so a name can be a prefix of a longer word;
    requiring whitespace-delimited equality would count correctly-copied names
    as misses.

    Args:
        record: The dataset record, whose ``prompt`` is the source of truth.
        parsed: The parse of the generation.

    Returns:
        ``chain_names``, ``chain_names_copied`` and ``copy_rate`` -- the last
        None when the chain was empty, since a rate over nothing is not zero,
        it is undefined.  Callers average only the non-None rows.
    """
    prompt = record["prompt"]
    names = parsed.chain
    if not names:
        return {"chain_names": 0, "chain_names_copied": 0, "copy_rate": None}
    copied = sum(1 for name in names if name in prompt)
    return {"chain_names": len(names), "chain_names_copied": copied,
            "copy_rate": copied / len(names)}


def score_record(record: dict, generated: str, parsed: ParsedCompletion,
                 labels: dict, generated_tokens: int, cap: int) -> dict:
    """Score one generation against one record.

    Args:
        record: The dataset record.
        generated: The decoded completion.
        parsed: Its parse.
        labels: The language's affirmative/negative labels.
        generated_tokens: How many tokens were produced.
        cap: The generation cap, to detect non-termination.

    Returns:
        A flat score dict, one row per generation.
    """
    query = record["query"]
    gold_answer = record["answer"]
    entailed = entailed_label(parsed.chain, query, labels)

    terminated = generated_tokens < cap
    format_valid = bool(parsed.has_chain_marker and parsed.has_answer_cue
                        and terminated)

    exact = parsed.label == gold_answer
    chain_correct = parsed.chain == record["true_order"]
    follows = (None if entailed is None or parsed.label is None
               else parsed.label == entailed)

    row = {
        "id": record["id"],
        "slice": record["slice"],
        "task_family": record["task_family"],
        "question_type": record["question_type"],
        "question_variant": record["question_variant"],
        "domain": record["domain"],
        "depth": record["depth"],
        "query_kind": query["kind"],
        "gold_answer": gold_answer,
        "predicted_answer": parsed.label,
        "em": bool(exact),
        "chain_correct": bool(chain_correct),
        "answer_follows_own_chain": follows,
        "format_valid": format_valid,
        "terminated": terminated,
        "generated_tokens": generated_tokens,
        "unparseable": bool(parsed.unparseable),
        "unparseable_reasons": parsed.unparseable,
        **copy_rate(record, parsed),
    }

    if query["kind"] == "bound":
        # A model that echoes the question's polarity answers "yes" whatever
        # the chain says: answering affirmatively to "is X less than V?" IS
        # asserting the direction the question proposed.  So the echo test
        # reduces to whether the label was affirmative.
        row["question_direction"] = query["direction"]
        row["answered_affirmative"] = parsed.label == labels["affirmative"]
        row["polarity_echo"] = row["answered_affirmative"]
        # The bound step the model wrote, and whether it agrees with its chain.
        row["bound_symbol"] = parsed.bound_symbol
        row["bound_step_matches_chain"] = _bound_step_agrees(parsed, query)

    return row


def _bound_step_agrees(parsed: ParsedCompletion, query: dict) -> bool | None:
    """Whether the emitted bound step agrees with the emitted chain.

    Args:
        parsed: The parse.
        query: The record's query block.

    Returns:
        True/False, or None if the bound step or the entities are missing.
    """
    if not parsed.bound_symbol or not parsed.chain:
        return None
    entity = parsed.bound_entity
    anchor = parsed.bound_anchor
    if entity not in parsed.chain or anchor not in parsed.chain:
        return None
    above = parsed.chain.index(entity) < parsed.chain.index(anchor)
    return (parsed.bound_symbol == ">") == above


def row_chance(row: dict) -> float:
    """Probability of guessing this question's answer without reasoning.

    Exact match is not comparable across slices without this.  A bound or
    relational-pairwise question has a two-element answer space, so 50% is what
    a coin scores; an extremum question over ``d`` entities gives ``1/d``; an
    ordering question has to name a whole permutation, so ``1/d!``.  A slice
    whose questions are mostly boolean therefore shows a high exact match for
    reasons that have nothing to do with reasoning, and reporting EM beside its
    own chance baseline is what keeps that visible.

    Args:
        row: One score row.

    Returns:
        The chance probability for that question.
    """
    kind = row["query_kind"]
    depth = max(2, row["depth"])
    if kind in ("pairwise", "bound"):
        return 0.5
    if kind == "extremum":
        return 1.0 / depth
    if kind == "ordering":
        return 1.0 / math.factorial(depth)
    return 0.0


def chance_baseline(rows: Sequence[dict]) -> float:
    """Mean chance probability across a set of questions.

    Args:
        rows: Score rows.

    Returns:
        The slice's chance exact match.
    """
    return (sum(row_chance(r) for r in rows) / len(rows)) if rows else 0.0


def rate(rows: Sequence[dict], key: str) -> float:
    """Mean of a boolean field, ignoring rows where it is None.

    Args:
        rows: Score rows.
        key: Field name.

    Returns:
        The rate, or 0.0 if nothing is defined.
    """
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in values if v) / len(values) if values else 0.0


def mean_copy_rate(rows: Sequence[dict]) -> float | None:
    """Mean copy-rate over the rows where it is defined.

    Rows whose generation produced no chain have no copy-rate -- there were no
    names to copy -- and are excluded rather than scored as 0.0, which would
    conflate "invented every name" with "emitted nothing to check".  The count
    of rows actually scored is reported alongside so a mean over very few
    parseable generations cannot be read as a confident number.

    Args:
        rows: Score rows.

    Returns:
        The mean, or None if no row carried a chain.
    """
    scored = [r["copy_rate"] for r in rows if r.get("copy_rate") is not None]
    return sum(scored) / len(scored) if scored else None


def summarize(rows: Sequence[dict]) -> dict:
    """Headline rates for a set of generations.

    Args:
        rows: Score rows.

    Returns:
        Counts and rates, including the 2x2 of §2.4.
    """
    total = len(rows)
    if not total:
        return {"n": 0}

    # The 2x2.  Rows whose entailment could not be determined (a chain missing
    # the queried entities) are counted separately rather than folded into a
    # cell, since assigning them either way would be a guess.
    cells = Counter()
    undetermined = 0
    for row in rows:
        follows = row["answer_follows_own_chain"]
        if follows is None:
            undetermined += 1
            continue
        cells[(row["chain_correct"], follows)] += 1

    return {
        "n": total,
        "em": rate(rows, "em"),
        "em_correct": sum(1 for r in rows if r["em"]),
        "chance_em": chance_baseline(rows),
        "em_above_chance": rate(rows, "em") - chance_baseline(rows),
        "chain_correct": rate(rows, "chain_correct"),
        "answer_follows_own_chain": rate(rows, "answer_follows_own_chain"),
        "format_valid": rate(rows, "format_valid"),
        "copy_rate": mean_copy_rate(rows),
        "copy_rate_scored": sum(1 for r in rows if r.get("copy_rate") is not None),
        "terminated": rate(rows, "terminated"),
        "unparseable": rate(rows, "unparseable"),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / total,
        "matrix": {
            "chain_correct_answer_follows": cells[(True, True)],
            "chain_correct_answer_contradicts": cells[(True, False)],
            "chain_wrong_answer_follows": cells[(False, True)],
            "chain_wrong_answer_contradicts": cells[(False, False)],
            "undetermined": undetermined,
        },
        # The gap IS the bug's footprint: reasoning right, reading it wrong.
        "self_contradiction_rate": cells[(True, False)] / total,
        "chain_correct_not_em": sum(
            1 for r in rows if r["chain_correct"] and not r["em"]) / total,
    }


def group_by(rows: Sequence[dict], *keys: str) -> dict:
    """Group score rows by one or more fields.

    Args:
        rows: Score rows.
        keys: Field names to group on.

    Returns:
        Tuple key to its rows.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return dict(grouped)


def breakdown(rows: Sequence[dict], *keys: str) -> dict:
    """Summarize each group of rows.

    Args:
        rows: Score rows.
        keys: Field names to group on.

    Returns:
        A string key (joined by ``|``) to its summary.
    """
    return {"|".join(str(part) for part in key): summarize(group)
            for key, group in sorted(group_by(rows, *keys).items(),
                                     key=lambda kv: str(kv[0]))}
