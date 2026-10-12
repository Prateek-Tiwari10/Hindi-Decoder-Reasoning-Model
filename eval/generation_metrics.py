"""Part 6 -- sampling continuations and scoring them.

THE TWO DECODING RULES, WRITTEN OUT
-----------------------------------
Both operate on the logits for the *next* token, i.e. the final position's row
of the model's output, shape ``(B, V)``.

**Greedy** takes ``argmax`` over the vocabulary.  Deterministic: the same prefix
always yields the same continuation.  It maximises the probability of each token
in isolation, which is not the same as maximising the probability of the
sequence -- and it is why greedy decoding falls into loops.  Once a phrase
becomes locally most-likely it stays locally most-likely, so the model repeats it
forever.

**Temperature** divides the logits by ``T`` before the softmax::

    p = softmax(logits / T)

Because softmax is shift-invariant but not scale-invariant, dividing by ``T``
rescales the *gaps* between logits:

  * ``T < 1`` widens the relative gaps -> distribution sharpens -> more like
    greedy.  ``T -> 0`` is exactly greedy.
  * ``T = 1`` leaves the model's own distribution untouched.
  * ``T > 1`` compresses the gaps -> distribution flattens -> rare tokens gain
    probability -> more diverse, less coherent.  ``T -> infinity`` is uniform
    sampling over the vocabulary.

Temperature is applied to the logits, never to the probabilities: softmaxing
first and then exponentiating would not renormalise correctly.

WHY chrF IS THE METRIC TO TRUST HERE
------------------------------------
BLEU-4 counts matching word 4-grams.  For open-ended generation that is close to
meaningless -- a continuation can be fluent, on-topic and completely correct
while sharing almost no 4-gram with the one reference we happen to have.  BLEU
is built for translation, where the output is heavily constrained by the input.

chrF scores *character* n-gram overlap, which matters twice over for Devanagari:
it is tokenizer-agnostic, so models with different tokenizers would be measured
on the same footing; and it gives partial credit for morphological near
misses.  Hindi is heavily inflected, so a generation that gets the
stem right and the case ending wrong is much closer to correct than a word-level
metric will admit.  chrF sees that; BLEU scores it zero.

All three are reported because the brief asks for them, but the report should
lead with chrF and say why.

THE DIVERSITY DIAGNOSTICS
-------------------------
Repetition rate and Distinct-n measure a failure mode the overlap metrics miss.
A model that emits the same sentence forever can score respectably on chrF
against a reference sharing common function words, so degeneracy needs its own
measurement.  The expected pattern -- and a check that the wiring is right -- is
that repetition falls and Distinct-n rises as temperature increases.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.config import ModelConfig, Paths
from data.token_stream import TokenStream
from eval.lm_metrics import load_trained
from model.common import generate

LOGGER = logging.getLogger("generate")

#: Decoding settings from the plan.
TEMPERATURES = (0.5, 1.0, 1.5)


def repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """Fraction of n-grams that are not unique.

    ``1 - distinct/total``: 0.0 means every n-gram appears once, and a value
    near 1.0 means the text is a loop.

    Args:
        token_ids: Generated tokens.
        n: n-gram size.

    Returns:
        Repetition rate in [0, 1], or 0.0 if the text is shorter than ``n``.
    """
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(sequences: list[list[int]], n: int) -> float:
    """Distinct-n over a corpus of generations.

    Unique n-grams divided by total n-grams, pooled across all samples so it
    also catches a model that produces different-looking samples which are
    nonetheless drawn from a tiny repertoire.

    Args:
        sequences: One token list per generation.
        n: n-gram size.

    Returns:
        Distinct-n in [0, 1].
    """
    total, unique = 0, set()
    for seq in sequences:
        if len(seq) < n:
            continue
        grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def build_prompts(stream: TokenStream, tokenizer, n_prompts: int,
                  prefix_len: int, continuation_len: int,
                  seed: int = 0) -> list[dict]:
    """Cut prefix/reference pairs from a held-out split.

    Prefixes come from the *test* stream, so the model has never seen them.
    The reference is the text that genuinely followed, which is what the
    overlap metrics score against.

    Args:
        stream: Held-out token stream.
        tokenizer: SentencePieceProcessor for decoding.
        n_prompts: How many pairs.
        prefix_len: Prompt length in tokens.
        continuation_len: Reference length in tokens.
        seed: RNG seed for offset selection.

    Returns:
        One dict per prompt with token ids and decoded text.
    """
    rng = np.random.default_rng(seed)
    span = prefix_len + continuation_len
    starts = rng.integers(0, len(stream) - span - 1, size=n_prompts)

    prompts = []
    for start in starts:
        block = np.asarray(stream.tokens[start:start + span], dtype=np.int64)
        prefix, reference = block[:prefix_len], block[prefix_len:]
        prompts.append({
            "prefix_ids": prefix.tolist(),
            "prefix_text": tokenizer.decode(prefix.tolist()),
            "reference_ids": reference.tolist(),
            "reference_text": tokenizer.decode(reference.tolist()),
        })
    return prompts


def score_generations(hypotheses: list[str], references: list[str]) -> dict:
    """Corpus-level BLEU-4, chrF and ROUGE-L.

    BLEU and chrF come from ``sacrebleu``, the standard implementation, so the
    numbers are comparable with published work rather than being a hand-rolled
    variant. ROUGE-L is computed here from the LCS, which is its definition.

    Args:
        hypotheses: Generated continuations.
        references: The true continuations.

    Returns:
        The three scores.
    """
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    # Devanagari has no whitespace-delimited concept BLEU's default tokenizer
    # handles well; chrF sidesteps the question by working on characters.
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    return {
        "bleu4": bleu.score,
        "chrf2": chrf.score,
        "rouge_l": corpus_rouge_l(hypotheses, references),
    }


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        LCS length.
    """
    # Rolling two-row table: the full (len(a), len(b)) matrix is unnecessary
    # since each row depends only on the previous one.
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            current[j] = (previous[j - 1] + 1 if token_a == token_b
                          else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def corpus_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Corpus ROUGE-L as the mean per-sample F1 over the LCS.

    Args:
        hypotheses: Generated continuations.
        references: True continuations.

    Returns:
        Mean ROUGE-L F1 in [0, 100].
    """
    scores = []
    for hypothesis, reference in zip(hypotheses, references):
        h, r = hypothesis.split(), reference.split()
        if not h or not r:
            scores.append(0.0)
            continue
        lcs = lcs_length(h, r)
        if lcs == 0:
            scores.append(0.0)
            continue
        precision, recall = lcs / len(h), lcs / len(r)
        scores.append(2 * precision * recall / (precision + recall))
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def run_setting(model, prompts: list[dict], tokenizer, device: str,
                temperature: float | None, max_new_tokens: int,
                batch_size: int, seed: int) -> dict:
    """Generate and score one decoding setting.

    Args:
        model: Trained model.
        prompts: Prefix/reference pairs.
        tokenizer: SentencePieceProcessor.
        device: Torch device.
        temperature: ``None`` for greedy.
        max_new_tokens: Continuation length.
        batch_size: Prompts per batch.
        seed: RNG seed, so samples are reproducible.

    Returns:
        Metrics, diagnostics and a few qualitative samples.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    hypotheses, hypothesis_ids = [], []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        prefix = torch.tensor([p["prefix_ids"] for p in batch], device=device)
        out = generate(model, prefix, max_new_tokens, temperature,
                       generator=generator)
        # Keep only what was generated; the prefix is not the model's output.
        new = out[:, prefix.shape[1]:].cpu().numpy()
        for row in new:
            hypothesis_ids.append(row.tolist())
            hypotheses.append(tokenizer.decode(row.tolist()))

    references = [p["reference_text"] for p in prompts]
    metrics = score_generations(hypotheses, references)

    label = "greedy" if temperature is None else f"temperature={temperature}"
    return {
        "setting": label,
        "temperature": temperature,
        **metrics,
        "repetition_rate_4gram": float(np.mean(
            [repetition_rate(ids, 4) for ids in hypothesis_ids])),
        "distinct_1": distinct_n(hypothesis_ids, 1),
        "distinct_2": distinct_n(hypothesis_ids, 2),
        "samples": [
            {"prefix": p["prefix_text"][-160:],
             "generated": h[:320],
             "reference": p["reference_text"][:320]}
            for p, h in list(zip(prompts, hypotheses))[:3]],
    }


def run(language: str, device: str, n_prompts: int, prefix_len: int,
        max_new_tokens: int, batch_size: int, seed: int,
        root: Path = Path("."), variant: str = "16k",
        checkpoint: Path | None = None) -> dict:
    """Evaluate one model across all decoding settings.

    Args:
        language: ``"hindi"`` (the only language).
        device: Torch device.
        n_prompts: Number of prefix/reference pairs.
        prefix_len: Prompt length in tokens.
        max_new_tokens: Continuation length in tokens.
        batch_size: Prompts per batch.
        seed: RNG seed.
        root: Repository root.
        variant: Architecture variant, selecting the checkpoint, tokenizer and
            test bin.
        checkpoint: Override the checkpoint; defaults to the variant's
            ``final.pt``.

    Returns:
        One entry per decoding setting.
    """
    import sentencepiece as spm

    paths = Paths.for_variant(language, root, variant)
    model, config = load_trained(language, checkpoint or paths.final_model,
                                 device, root, variant)
    tokenizer = spm.SentencePieceProcessor(model_file=str(paths.tokenizer))
    stream = TokenStream(paths.split_bin("test"), config.context_len)

    prompts = build_prompts(stream, tokenizer, n_prompts, prefix_len,
                            max_new_tokens, seed)

    settings = []
    for temperature in (None,) + TEMPERATURES:
        label = "greedy" if temperature is None else f"T={temperature}"
        LOGGER.info("  %s: generating %d x %d tokens ...",
                    label, n_prompts, max_new_tokens)
        settings.append(run_setting(model, prompts, tokenizer, device,
                                    temperature, max_new_tokens, batch_size,
                                    seed))
    return {"language": language, "n_prompts": n_prompts,
            "prefix_tokens": prefix_len, "generated_tokens": max_new_tokens,
            "settings": settings}


def main() -> None:
    """Generate and score for both models."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--language", choices=("hindi",), default="hindi")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--prefix-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--variant", default="16k",
                        help='Variant: "16k" (GPT baseline) or "16k_nopos" '
                             '(no-positional ablation).')
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained checkpoint to load. Defaults to the variant's final.pt. Needed because a reduced-budget run produces last.pt / best.pt rather than a promoted final.")
    parser.add_argument("--report", type=Path,
                        default=Path("reports/gpt_pretrain/generation.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    languages = ("hindi",)

    reports = {}
    for language in languages:
        LOGGER.info("=== %s ===", language)
        reports[language] = run(language, device, args.n_prompts,
                                args.prefix_len, args.max_new_tokens,
                                args.batch_size, args.seed, args.root,
                                args.variant, args.checkpoint)

    LOGGER.info("\n%-8s %-16s %7s %7s %8s %7s %7s %7s",
                "lang", "setting", "BLEU4", "chrF2", "ROUGE-L",
                "rep4", "dist1", "dist2")
    LOGGER.info("-" * 78)
    for language, report in reports.items():
        for s in report["settings"]:
            LOGGER.info("%-8s %-16s %7.2f %7.2f %8.2f %7.3f %7.3f %7.3f",
                        language, s["setting"], s["bleu4"], s["chrf2"],
                        s["rouge_l"], s["repetition_rate_4gram"],
                        s["distinct_1"], s["distinct_2"])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
