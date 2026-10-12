"""Train the filter used to keep only Hindi documents (Hindi-vs-Nepali LID).

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Training data comes from Wikipedia in each language, which is the right choice
for three reasons: the labels are reliable (each wiki is edited in its own
language), it is completely independent of the news crawl the classifier is
applied to, and it is already counted as a *downloaded* source, so using it here
does not disturb the manual/downloaded accounting.

The script does four things:

1. Sample articles from both wikis via the HuggingFace datasets-server HTTP API,
   caching them to disk so re-runs cost nothing.
2. Train the character n-gram model on a training split.
3. **Calibrate the decision thresholds** on a held-out split by sweeping the
   margin threshold and choosing the smallest one that reaches the target
   precision.  Thresholds are fitted, not guessed.
4. Report held-out precision/recall/rejection rates for the Phase 1 statistics.

Usage::

    python -m data.pipeline.train_lid --n-docs 6000 --out data/pipeline/models/lid_hi_ne.json

The target is precision, not accuracy.  Contaminating a corpus is far more
damaging to this project than discarding a usable document, so the calibration
optimizes for precision and simply reports whatever recall results.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from data.pipeline.lid import (
    AMBIGUOUS,
    HINDI,
    NEPALI,
    LanguageIdentifier,
    LidConfig,
    evaluate,
    train,
)
from data.pipeline.normalize import DevanagariNormalizer, NormalizerConfig, devanagari_ratio

LOGGER = logging.getLogger("train_lid")

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_WIKI_CONFIG = {HINDI: "20231101.hi", NEPALI: "20231101.ne"}
_USER_AGENT = "LMA-course-research/1.0 (academic corpus construction)"

#: Rows per API call. The endpoint caps the page size, and 100 keeps each
#: request small enough to retry cheaply.
_PAGE_SIZE = 100


def _get_json(url: str, retries: int = 4) -> dict:
    """GET a JSON endpoint with backoff on rate limiting.

    Args:
        url: Absolute URL.
        retries: Attempts before giving up.

    Returns:
        Decoded JSON payload.

    Raises:
        RuntimeError: If every attempt fails.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last = exc
            # 429/5xx are transient; back off and retry.
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last


def fetch_wikipedia(language: str, n_docs: int, cache_dir: Path) -> list[str]:
    """Sample Wikipedia articles for one language, caching to disk.

    Articles are drawn from evenly spaced offsets rather than from the start of
    the dump, because Wikipedia dumps are ordered by page id and the earliest
    pages are disproportionately stubs and administrative pages.  Spreading the
    sample gives a more representative mix of article lengths and topics.

    Args:
        language: ``"hi"`` or ``"ne"``.
        n_docs: Target number of articles.
        cache_dir: Directory holding the JSON cache.

    Returns:
        Article texts, filtered to those that are long enough and predominantly
        Devanagari.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"wikipedia_{language}_{n_docs}.json"
    if cache_path.exists():
        LOGGER.info("%s: loading %d cached articles", language, n_docs)
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    config = _WIKI_CONFIG[language]
    base = f"{_ROWS_API}?dataset={urllib.parse.quote('wikimedia/wikipedia')}&config={config}&split=train"
    total = _get_json(f"{base}&offset=0&length=1")["num_rows_total"]
    LOGGER.info("%s: wiki has %d articles, sampling %d", language, total, n_docs)

    n_pages = max(1, n_docs // _PAGE_SIZE)
    stride = max(1, total // n_pages)

    docs: list[str] = []
    for page in range(n_pages):
        offset = min(page * stride, max(0, total - _PAGE_SIZE))
        try:
            payload = _get_json(f"{base}&offset={offset}&length={_PAGE_SIZE}")
        except Exception as exc:
            LOGGER.warning("%s: offset %d failed: %s", language, offset, exc)
            continue
        for row in payload.get("rows", []):
            text = row.get("row", {}).get("text", "")
            # Wikipedia contains disambiguation pages, stubs and list pages that
            # carry almost no running prose; they would teach the classifier
            # nothing and dilute the n-gram statistics.
            if len(text) >= 400 and devanagari_ratio(text) >= 0.6:
                docs.append(text)
        if len(docs) >= n_docs:
            break
        time.sleep(0.2)  # stay well inside the public endpoint's rate limit

    docs = docs[:n_docs]
    LOGGER.info("%s: kept %d articles", language, len(docs))
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False)
    return docs


def calibrate_threshold(
    identifier: LanguageIdentifier,
    hindi_docs: list[str],
    nepali_docs: list[str],
    target_precision: float = 0.995,
) -> tuple[float, dict]:
    """Fit the margin threshold that reaches a target held-out precision.

    Sweeps ``min_abs_logratio`` upward.  Raising it makes the classifier abstain
    on more documents, which trades recall for precision -- the trade this
    project wants, since corpus volume is abundant and contamination is not
    recoverable once it is baked into a trained model.

    Args:
        identifier: Classifier to calibrate (mutated during the sweep, restored
            before returning).
        hindi_docs: Held-out Hindi documents.
        nepali_docs: Held-out Nepali documents.
        target_precision: Minimum acceptable precision for both classes.

    Returns:
        Tuple of ``(chosen_threshold, sweep_table)`` where the table maps each
        tried threshold to its measured metrics.
    """
    original = identifier.config.min_abs_logratio
    sweep: dict[str, dict] = {}
    chosen = None

    for threshold in (0.0, 0.002, 0.005, 0.010, 0.020, 0.030, 0.050, 0.080):
        identifier.config.min_abs_logratio = threshold
        metrics = evaluate(identifier, hindi_docs, nepali_docs)
        worst_precision = min(metrics[HINDI]["precision"], metrics[NEPALI]["precision"])
        mean_recall = (metrics[HINDI]["recall"] + metrics[NEPALI]["recall"]) / 2
        leaked = (
            metrics[HINDI]["leaked_into_other_corpus"]
            + metrics[NEPALI]["leaked_into_other_corpus"]
        )
        sweep[f"{threshold:.3f}"] = {
            "precision_min": round(worst_precision, 5),
            "recall_mean": round(mean_recall, 5),
            "leaked_docs": leaked,
        }
        LOGGER.info(
            "  threshold=%.3f  precision=%.4f  recall=%.4f  leaked=%d",
            threshold, worst_precision, mean_recall, leaked,
        )
        if chosen is None and worst_precision >= target_precision:
            chosen = threshold

    if chosen is None:
        # No threshold reached the target: keep the strictest tried value and
        # let the caller see it in the report rather than silently accepting
        # worse precision than requested.
        chosen = 0.080
        LOGGER.warning(
            "no threshold reached precision %.3f; falling back to %.3f",
            target_precision, chosen,
        )

    identifier.config.min_abs_logratio = original
    return chosen, sweep


#: Document lengths (in Devanagari words) the classifier is evaluated at.
#: Full Wikipedia articles are an unrealistically easy test -- the crawl produces
#: mostly short news text, and truncated snippets are where a Hindi/Nepali
#: classifier actually fails. Thresholds are fitted on the *shortest* stratum so
#: they hold in the worst case rather than the average one.
LENGTH_STRATA: tuple[int | None, ...] = (30, 60, 120, 300, None)


def truncate_to_words(text: str, n_words: int | None) -> str:
    """Truncate text to roughly the first ``n_words`` Devanagari words.

    Args:
        text: Document text.
        n_words: Word budget, or ``None`` to return the text unchanged.

    Returns:
        The truncated text.
    """
    if n_words is None:
        return text
    words = text.split()
    return " ".join(words[:n_words])


def evaluate_by_length(
    identifier: LanguageIdentifier,
    hindi_docs: list[str],
    nepali_docs: list[str],
) -> dict:
    """Measure precision and recall separately at each document length.

    This is the evaluation that matters.  Reporting only full-article accuracy
    would overstate the classifier badly, because the crawl is dominated by short
    documents and short text is exactly where two closely related languages
    become hard to separate.

    Args:
        identifier: Trained and configured classifier.
        hindi_docs: Held-out Hindi documents.
        nepali_docs: Held-out Nepali documents.

    Returns:
        Mapping from stratum label to that stratum's metrics.
    """
    out: dict = {}
    for n_words in LENGTH_STRATA:
        label = f"{n_words}w" if n_words else "full"
        metrics = evaluate(
            identifier,
            [truncate_to_words(d, n_words) for d in hindi_docs],
            [truncate_to_words(d, n_words) for d in nepali_docs],
        )
        out[label] = {
            "precision_hi": round(metrics[HINDI]["precision"], 5),
            "precision_ne": round(metrics[NEPALI]["precision"], 5),
            "recall_hi": round(metrics[HINDI]["recall"], 5),
            "recall_ne": round(metrics[NEPALI]["recall"], 5),
            "leaked_docs": (
                metrics[HINDI]["leaked_into_other_corpus"]
                + metrics[NEPALI]["leaked_into_other_corpus"]
            ),
            "rejected_docs": (
                metrics[HINDI]["rejected_as_ambiguous"]
                + metrics[NEPALI]["rejected_as_ambiguous"]
            ),
        }
    return out


def margin_statistics(
    identifier: LanguageIdentifier, docs: list[str], label: str
) -> dict:
    """Summarize the score margin distribution for one language.

    Reported in the Phase 1 write-up as evidence that the two languages really
    are separable by this feature set, and used to sanity-check the confidence
    scale.

    Args:
        identifier: Trained classifier.
        docs: Documents of a single known language.
        label: The known language of ``docs``.

    Returns:
        Percentile summary of the signed per-n-gram margin.
    """
    margins = [identifier.predict(d).ngram_logratio for d in docs]
    margins.sort()
    if not margins:
        return {}
    def pct(p: float) -> float:
        """Return the p-th percentile of the sorted margins."""
        return round(margins[min(len(margins) - 1, int(p * len(margins)))], 5)
    return {
        "language": label,
        "n": len(margins),
        "p05": pct(0.05), "p25": pct(0.25), "median": pct(0.50),
        "p75": pct(0.75), "p95": pct(0.95),
        "mean": round(statistics.fmean(margins), 5),
    }


def main() -> None:
    """Fetch data, train, calibrate, evaluate and save the LID model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-docs", type=int, default=6000,
                        help="articles to sample per language")
    parser.add_argument("--order", type=int, default=4,
                        help="character n-gram order")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--target-precision", type=float, default=0.995)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/pipeline/models/cache"))
    parser.add_argument("--out", type=Path, default=Path("data/pipeline/models/lid_hi_ne.json"))
    parser.add_argument("--report", type=Path,
                        default=Path(f"reports/data/lid_evaluation.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    raw = {lang: fetch_wikipedia(lang, args.n_docs, args.cache_dir)
           for lang in (HINDI, NEPALI)}

    # Normalize exactly as the cleaning pipeline will, so the classifier sees
    # training text in the same form as the documents it will judge.
    normalizers = {
        HINDI: DevanagariNormalizer(NormalizerConfig.for_hindi()),
        NEPALI: DevanagariNormalizer(NormalizerConfig.for_nepali()),
    }
    docs = {lang: [normalizers[lang](d) for d in raw[lang]] for lang in raw}

    splits = {}
    for lang, items in docs.items():
        cut = int(len(items) * (1 - args.test_fraction))
        splits[lang] = (items[:cut], items[cut:])
        LOGGER.info("%s: train=%d test=%d", lang, cut, len(items) - cut)

    LOGGER.info("training %d-gram model...", args.order)
    model = train(splits[HINDI][0], splits[NEPALI][0], order=args.order)
    LOGGER.info("vocabulary: %d distinct n-grams", model.vocabulary_size)

    identifier = LanguageIdentifier(model, LidConfig(order=args.order))

    LOGGER.info("margin distribution on held-out text:")
    margins = {
        lang: margin_statistics(identifier, splits[lang][1], lang)
        for lang in (HINDI, NEPALI)
    }
    for lang, stats in margins.items():
        LOGGER.info("  %s: median=%+.4f p05=%+.4f p95=%+.4f",
                    lang, stats["median"], stats["p05"], stats["p95"])

    # Calibrate on the SHORTEST stratum, not on full articles. Full articles are
    # separable by a mile, so calibrating on them would pick a threshold that
    # looks perfect in the report and does nothing on the crawl.
    short = min(w for w in LENGTH_STRATA if w is not None)
    LOGGER.info(
        "calibrating margin threshold on the %d-word stratum "
        "(target precision %.3f):", short, args.target_precision,
    )
    threshold, sweep = calibrate_threshold(
        identifier,
        [truncate_to_words(d, short) for d in splits[HINDI][1]],
        [truncate_to_words(d, short) for d in splits[NEPALI][1]],
        args.target_precision,
    )
    identifier.config.min_abs_logratio = threshold
    LOGGER.info("chosen min_abs_logratio = %.3f", threshold)

    LOGGER.info("held-out metrics by document length:")
    by_length = evaluate_by_length(identifier, splits[HINDI][1], splits[NEPALI][1])
    LOGGER.info("  %-6s %9s %9s %9s %9s %8s %9s",
                "len", "prec_hi", "prec_ne", "rec_hi", "rec_ne", "leaked", "rejected")
    for label, m in by_length.items():
        LOGGER.info("  %-6s %9.4f %9.4f %9.4f %9.4f %8d %9d",
                    label, m["precision_hi"], m["precision_ne"],
                    m["recall_hi"], m["recall_ne"], m["leaked_docs"],
                    m["rejected_docs"])

    final = evaluate(identifier, splits[HINDI][1], splits[NEPALI][1])
    LOGGER.info("-" * 62)
    for lang in (HINDI, NEPALI):
        metrics = final[lang]
        LOGGER.info(
            "%s  precision=%.4f  recall=%.4f  rejected=%d  leaked=%d",
            lang, metrics["precision"], metrics["recall"],
            metrics["rejected_as_ambiguous"], metrics["leaked_into_other_corpus"],
        )

    model.save(args.out)
    LOGGER.info("model -> %s", args.out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "order": args.order,
                "n_docs_per_language": args.n_docs,
                "vocabulary_size": model.vocabulary_size,
                "n_train_docs": model.n_train_docs,
                "chosen_min_abs_logratio": threshold,
                "target_precision": args.target_precision,
                "calibrated_on_stratum_words": short,
                "threshold_sweep": sweep,
                "metrics_by_document_length": by_length,
                "margin_distribution": margins,
                "held_out_metrics": {
                    lang: final[lang] for lang in (HINDI, NEPALI)
                },
                "confusion": {
                    outer: {inner: final["confusion"][outer][inner]
                            for inner in (HINDI, NEPALI, AMBIGUOUS)}
                    for outer in (HINDI, NEPALI)
                },
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
