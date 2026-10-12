"""Phase 3 generation quality — BLEU / chrF / ROUGE-L, diversity, repetition.

WHY THESE METRICS APPEAR HERE AND NOT ONLY AS EXACT MATCH
----------------------------------------------------------
Exact match scores the final answer token.  It says nothing about the reasoning
chain the model wrote to get there, which is most of the output and all of the
interesting part.  A model can reach `हाँ` by luck with a mangled chain, and
§7.2 of the Part 3A report showed exactly that happening.

These are the same metrics Phase 2 used on free generation, re-applied to the
finetuned models against the gold completion:

  * **chrF2** — character n-gram F-score.  The metric to trust on Devanagari:
    it does not depend on whitespace tokenisation, which Hindi does
    not use the way BLEU's default tokeniser assumes.
  * **BLEU-4** — word 4-gram precision.  Reported because it is standard, but
    it is the weaker signal here for the reason above.
  * **ROUGE-L** — longest common subsequence, computed from its definition.
    Rewards getting the chain's *order* right even when tokens differ.
  * **Distinct-1 / Distinct-2** — unique unigrams and bigrams over the pooled
    corpus.  Catches a model drawing from a tiny repertoire.
  * **Repetition rate** — fraction of 4-grams that are not unique.  Catches the
    degenerate loop the pretrained base falls into.

REFERENCES COME FROM THE DATASET, NOT THE RAW EVAL FILE
-------------------------------------------------------
The scored rows carry the generation and the gold *answer*, but not the gold
*completion*.  The completion is joined back from the corpus on the record id,
so the reference is the exact text the generator wrote for that item.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from eval.generation_metrics import distinct_n, repetition_rate, score_generations

LOGGER = logging.getLogger("eval.reasoning_generation")

LANGS = (("hindi", "hi"),)
#: Model tags whose raw rows (from ``eval.reasoning_eval``) are scored.
MODELS = ("ft_60k",)

#: Slice groups the report aggregates over, mirroring the scaling tables.
GROUPS = {
    "in-domain d2-3": ("rel_id_d23", "num_id_d23", "mix_id_d23"),
    "in-domain d4-6": ("rel_ood_d4", "rel_ood_d5", "rel_ood_d6"),
    "cross-domain d2-3": ("xdom_speed_id_d23", "xdom_strength_id_d23",
                          "xdom_diligence_id_d23"),
}

SLICE_DIR = {"in_domain": ("rel_", "num_", "mix_"), "cross_domain": ("xdom_",)}


def corpus_path(root: Path, code: str, slice_tag: str) -> Path:
    """Where the generator wrote this slice.

    Cross-domain slices are nested one level further, under the domain name,
    so the path is resolved by searching rather than by assuming a shape.
    """
    from model.config import Paths

    base = Paths.for_variant("hindi", root).finetune_data / "test" / code
    direct = base / "in_domain" / f"{slice_tag}.jsonl"
    if direct.exists():
        return direct
    matches = sorted(base.rglob(f"{slice_tag}.jsonl"))
    if not matches:
        raise SystemExit(f"no corpus file for slice {slice_tag!r} under {base}")
    return matches[0]


def references_for(root: Path, code: str, slice_tag: str) -> dict[str, str]:
    """Map record id -> gold completion for one slice."""
    path = corpus_path(root, code, slice_tag)
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["id"]] = row["completion"].strip()
    return out


def score_group(root: Path, code: str, language: str, model: str,
                slices: tuple[str, ...], tokenizer) -> dict | None:
    """Score one (model, slice-group) cell, or None if its files are absent."""
    hypotheses, refs, token_lists = [], [], []
    for slice_tag in slices:
        raw = (root / "reports" / "gpt_finetune" / "eval_raw"
               / f"{code}_{model}_{slice_tag}.jsonl")
        if not raw.exists():
            continue
        gold = references_for(root, code, slice_tag)
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                reference = gold.get(row["id"])
                if reference is None:
                    continue
                generated = (row.get("generated") or "").strip()
                hypotheses.append(generated)
                refs.append(reference)
                token_lists.append(tokenizer.encode(generated))
    if not hypotheses:
        return None

    overlap = score_generations(hypotheses, refs)
    rates = [repetition_rate(t, 4) for t in token_lists if len(t) >= 4]
    return {
        "n": len(hypotheses),
        "bleu4": overlap["bleu4"],
        "chrf2": overlap["chrf2"],
        "rouge_l": overlap["rouge_l"],
        "distinct_1": distinct_n(token_lists, 1),
        "distinct_2": distinct_n(token_lists, 2),
        "repetition_rate_4gram": sum(rates) / len(rates) if rates else 0.0,
        "mean_generated_tokens": sum(len(t) for t in token_lists) / len(token_lists),
    }


def main() -> None:
    """Score every (language, size, slice-group) cell."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_finetune/generation_metrics.json"))
    parser.add_argument("--models", nargs="+", default=None,
                        help="model tags with raw rows to score")
    args = parser.parse_args()

    import sentencepiece as spm
    from model.config import Paths

    root = args.root.resolve()
    payload: dict = defaultdict(dict)

    for language, code in LANGS:
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(Paths.for_variant(language, root, "16k").tokenizer))
        for model in (args.models or MODELS):
            for name, slices in GROUPS.items():
                cell = score_group(root, code, language, model, slices, tokenizer)
                if cell is None:
                    continue
                payload[language].setdefault(model, {})[name] = cell
                LOGGER.info("%s %s %-18s n=%4d chrF2 %5.2f BLEU %5.2f "
                            "ROUGE-L %5.2f  D1 %.3f D2 %.3f rep %.3f",
                            language[:2], model, name, cell["n"],
                            cell["chrf2"], cell["bleu4"], cell["rouge_l"],
                            cell["distinct_1"], cell["distinct_2"],
                            cell["repetition_rate_4gram"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
