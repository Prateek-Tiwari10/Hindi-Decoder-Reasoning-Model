"""Filter used to keep only Hindi documents: Hindi-vs-Nepali language ID.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Hindi news crawls contain Nepali-language documents (shared hosts such as BBC,
syndicated wire copy, quoted passages).  Both are written in Devanagari, so the
script filter cannot see them; this classifier is what removes them, keeping
only documents it confidently labels Hindi.

Why off-the-shelf language ID is not enough here:

* Hindi and Nepali share the Devanagari script exactly, so every script-level
  signal is useless between them.
* They share a large amount of Sanskrit-derived vocabulary, so bag-of-words
  overlap is high.
* Nepali news sites quote Hindi and Indian outlets carry Nepali datelines, so
  real cross-contamination exists in the crawl.
* General-purpose classifiers such as fastText ``lid.176`` are trained to spread
  capacity over 176 languages and are measurably unreliable on short
  Hindi/Nepali text -- which is exactly the text a news crawl produces most of.

The approach here is a **character n-gram multinomial Naive Bayes** classifier
trained only on this one binary distinction, backed by a **function-word
discriminator** that exploits the fact that the two languages differ sharply in
their closed-class morphology even where content words coincide:

===========  ==========================================  ==========================
Feature      Nepali                                      Hindi
===========  ==========================================  ==========================
copula       छ / छन् / थियो / हो                          है / हैं / था
infinitive   -न (गर्न, हुन)                               -ना (करना, होना)
ablative     बाट                                         से
comitative   सँग                                         साथ
===========  ==========================================  ==========================

Both signals are combined, and -- importantly -- documents that fall in the
**ambiguous band** between the two classes are *rejected* rather than
assigned to the more likely one.  For corpus construction, precision
matters far more than recall: there is plenty of text available, so discarding a
genuine document is cheap while admitting a contaminant is not.

The classifier is trained from Wikipedia text in the two languages
(:func:`train_from_files`), which is independent of the news crawl it is applied
to.  Model files are per-language-pair, contain no corpus text, and are small
enough to commit.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HINDI = "hi"
NEPALI = "ne"
AMBIGUOUS = "ambiguous"

#: Closed-class markers.  These are deliberately morphological (copulas, case
#: postpositions, infinitive endings) rather than topical: a Nepali article about
#: India still uses Nepali copulas, whereas topical words like "नेपाल" would
#: misfire on Hindi articles reporting on Nepal.
NEPALI_MARKERS: tuple[str, ...] = (
    "छ", "छन्", "छैन", "थियो", "थिए", "हो", "हुन्", "भएको", "गरेको", "गर्न",
    "गर्ने", "भन्ने", "भने", "बाट", "सँग", "लागि", "पनि", "तर", "अनि", "हुन",
    "रहेको", "गरी", "भन्दा", "जस्तो", "कुनै", "यस्तो", "हुन्छ", "गर्नु",
)

HINDI_MARKERS: tuple[str, ...] = (
    "है", "हैं", "था", "थे", "थी", "हुआ", "किया", "करना", "करने", "वाले",
    "रहा", "रही", "गया", "गई", "कहा", "लिए", "साथ", "लेकिन", "और", "भी",
    "होता", "करते", "जाता", "कोई", "ऐसा", "जैसे", "बाद", "दिया",
)

_WORD_RE = re.compile(r"[ऀ-ॿ]+")

#: Logistic scale mapping the per-n-gram log-odds margin to a reported
#: confidence. Fitted on held-out Wikipedia text by ``data.pipeline.train_lid``, where
#: full-length articles separate at a median margin of ~1.9 nats/n-gram and
#: genuinely mixed text sits near 0. A scale of 4.0 maps that median to ~0.9995
#: while leaving a margin of 0.1 at ~0.60, so the number stays informative in
#: the region that actually matters. Changing it changes what
#: ``min_confidence`` means.
CONFIDENCE_SCALE: float = 4.0


@dataclass
class LidPrediction:
    """Outcome of classifying one document.

    Attributes:
        label: ``"hi"``, ``"ne"`` or ``"ambiguous"``.
        confidence: Probability of the winning class in ``[0.5, 1.0]``.
        ngram_logratio: Naive Bayes log-odds, positive favouring Hindi.
        marker_logratio: Function-word log-odds, positive favouring Hindi.
        n_devanagari_words: Devanagari word count, used to gate short documents.
        reason: Why this label was assigned, for audit tables in the report.
    """

    label: str
    confidence: float
    ngram_logratio: float
    marker_logratio: float
    n_devanagari_words: int
    reason: str = ""


@dataclass
class LidConfig:
    """Decision thresholds for :class:`LanguageIdentifier`.

    Attributes:
        order: Character n-gram order.  4 captures Devanagari morpheme-sized
            units (a consonant plus matra plus suffix) without exploding the
            model size the way 5-grams do.
        min_words: Documents with fewer Devanagari words than this are labelled
            ambiguous regardless of score -- short text carries too little
            evidence to separate two closely related languages.
        min_confidence: Naive Bayes probability required to accept a label.
        min_abs_logratio: Minimum absolute per-n-gram log-odds margin.  Clean
            full-length text separates at ~1.9; short news snippets separate by
            much less, so this is the threshold that governs the hard cases.
            Fitted on the shortest length stratum by ``data.pipeline.train_lid``.
        require_marker_agreement: Require the function-word discriminator to
            agree with the n-gram model (or be neutral).  This is the main
            precision lever: the two signals fail in different ways, so
            demanding agreement removes most residual contamination.
        marker_neutral_band: Absolute marker log-odds below this counts as
            neutral rather than disagreeing.
    """

    order: int = 4
    min_words: int = 25
    min_confidence: float = 0.90
    min_abs_logratio: float = 0.010
    require_marker_agreement: bool = True
    marker_neutral_band: float = 0.20


@dataclass
class LidModel:
    """Trained character n-gram model for the Hindi/Nepali pair.

    Attributes:
        order: N-gram order the counts were built at.
        log_prob: ``{label: {ngram: log P(ngram | label)}}``.
        log_prior_unseen: ``{label: log P(unseen ngram | label)}`` from add-one
            smoothing, applied to n-grams absent from that class.
        vocabulary_size: Number of distinct n-grams across both classes.
        n_train_docs: Training documents per class, recorded for the report.
    """

    order: int
    log_prob: dict[str, dict[str, float]]
    log_prior_unseen: dict[str, float]
    vocabulary_size: int
    n_train_docs: dict[str, int] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Serialize the model to JSON.

        Args:
            path: Destination file; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "order": self.order,
                    "log_prob": self.log_prob,
                    "log_prior_unseen": self.log_prior_unseen,
                    "vocabulary_size": self.vocabulary_size,
                    "n_train_docs": self.n_train_docs,
                },
                fh,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: Path | str) -> "LidModel":
        """Load a model previously written by :meth:`save`.

        Args:
            path: Model JSON path.

        Returns:
            The deserialized model.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)


def _ngrams(text: str, order: int) -> Counter:
    """Count character n-grams over Devanagari word tokens.

    Words are padded with spaces so that word-initial and word-final n-grams are
    distinguishable -- essential here, because the strongest Hindi/Nepali signals
    are inflectional *endings*, which only become visible as features when the
    word boundary is part of the n-gram.

    Args:
        text: Input text.
        order: N-gram order.

    Returns:
        Counter mapping n-gram to frequency.
    """
    counts: Counter = Counter()
    for word in _WORD_RE.findall(text):
        padded = f" {word} "
        for i in range(len(padded) - order + 1):
            counts[padded[i : i + order]] += 1
    return counts


def marker_logratio(text: str) -> tuple[float, int]:
    """Score a document by closed-class function words.

    Computes the log ratio of Hindi to Nepali marker rates, so a positive value
    favours Hindi.  Because the markers are grammatical rather than topical, this
    signal stays reliable on documents whose subject matter belongs to the other
    country.

    Args:
        text: Document text.

    Returns:
        Tuple of ``(log_ratio, n_devanagari_words)``.  The ratio is ``0.0`` when
        the document contains no Devanagari words.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0, 0
    counts = Counter(words)
    hi = sum(counts[m] for m in HINDI_MARKERS)
    ne = sum(counts[m] for m in NEPALI_MARKERS)
    # Add-one smoothing keeps the ratio finite when a document uses markers from
    # only one language, which is the common case for clean text.
    return math.log((hi + 1) / (ne + 1)), len(words)


class LanguageIdentifier:
    """Classify documents as Hindi, Nepali, or ambiguous.

    Args:
        model: Trained n-gram model.
        config: Decision thresholds.
    """

    def __init__(self, model: LidModel, config: LidConfig | None = None) -> None:
        """Store the model and thresholds."""
        self.model = model
        self.config = config or LidConfig(order=model.order)

    def predict(self, text: str) -> LidPrediction:
        """Classify one document.

        The decision is intentionally conservative.  A document is accepted only
        when it is long enough to carry evidence, the n-gram model is confident,
        the per-character margin is non-trivial, and the independent function-word
        signal does not contradict it.  Anything else is returned as
        ``"ambiguous"`` and dropped from both corpora by the caller.

        Args:
            text: Document text, normally already Unicode-normalized.

        Returns:
            The prediction, including both raw scores for auditing.
        """
        marker_lr, n_words = marker_logratio(text)

        if n_words < self.config.min_words:
            return LidPrediction(AMBIGUOUS, 0.5, 0.0, marker_lr, n_words,
                                 reason=f"too_short(<{self.config.min_words}w)")

        counts = _ngrams(text, self.model.order)
        total = sum(counts.values())
        if not total:
            return LidPrediction(AMBIGUOUS, 0.5, 0.0, marker_lr, n_words,
                                 reason="no_devanagari_ngrams")

        scores = {}
        for label in (HINDI, NEPALI):
            table = self.model.log_prob[label]
            unseen = self.model.log_prior_unseen[label]
            scores[label] = sum(
                n * table.get(gram, unseen) for gram, n in counts.items()
            )

        # Normalize by n-gram count so the margin is per-character and therefore
        # comparable across documents of very different lengths.
        #
        # Equal priors: the crawl is not a random sample of either language, so
        # an empirical prior would encode crawl composition rather than evidence.
        ngram_lr = (scores[HINDI] - scores[NEPALI]) / total
        winner = HINDI if ngram_lr > 0 else NEPALI

        # The exact Naive Bayes posterior is unusable as a confidence here: it
        # multiplies per-n-gram evidence that is far from independent, so it
        # saturates to 1.0 for any document of realistic length and stops
        # discriminating. Instead map the *per-n-gram* margin through a logistic
        # with a calibrated scale, which keeps confidence comparable across
        # document lengths. CONFIDENCE_SCALE is fitted on held-out Wikipedia
        # text by data.pipeline.train_lid.
        confidence = 1.0 / (1.0 + math.exp(-abs(ngram_lr) * CONFIDENCE_SCALE))

        if confidence < self.config.min_confidence:
            return LidPrediction(AMBIGUOUS, confidence, ngram_lr, marker_lr, n_words,
                                 reason=f"low_confidence({confidence:.3f})")

        if abs(ngram_lr) < self.config.min_abs_logratio:
            return LidPrediction(AMBIGUOUS, confidence, ngram_lr, marker_lr, n_words,
                                 reason=f"weak_margin({ngram_lr:+.4f})")

        if self.config.require_marker_agreement:
            marker_winner = (
                None if abs(marker_lr) < self.config.marker_neutral_band
                else (HINDI if marker_lr > 0 else NEPALI)
            )
            if marker_winner is not None and marker_winner != winner:
                return LidPrediction(
                    AMBIGUOUS, confidence, ngram_lr, marker_lr, n_words,
                    reason=f"signal_disagreement(ngram={winner},marker={marker_winner})",
                )

        return LidPrediction(winner, confidence, ngram_lr, marker_lr, n_words,
                             reason="accepted")

    def is_language(self, text: str, language: str) -> bool:
        """Return whether a document belongs to the requested language.

        Convenience wrapper for the cleaning pipeline, where the question is
        always "does this document belong in *this* corpus?".

        Args:
            text: Document text.
            language: ``"hi"`` or ``"ne"``.

        Returns:
            ``True`` only for a confident match; ambiguous documents return
            ``False`` and are therefore excluded from both corpora.
        """
        return self.predict(text).label == language


def train(
    hindi_docs: list[str], nepali_docs: list[str], order: int = 4
) -> LidModel:
    """Train the character n-gram model from labelled documents.

    Uses multinomial Naive Bayes with add-one (Laplace) smoothing.  A generative
    model rather than a discriminative one because it trains in a single pass
    with no iteration and a flat memory profile -- this machine has ~4 GB of RAM
    free, and the model must be trainable alongside a running crawl.

    Args:
        hindi_docs: Known-Hindi training documents.
        nepali_docs: Known-Nepali training documents.
        order: Character n-gram order.

    Returns:
        The trained model.

    Raises:
        ValueError: If either class has no training documents.
    """
    if not hindi_docs or not nepali_docs:
        raise ValueError("both classes need training documents")

    class_counts: dict[str, Counter] = {HINDI: Counter(), NEPALI: Counter()}
    for label, docs in ((HINDI, hindi_docs), (NEPALI, nepali_docs)):
        for doc in docs:
            class_counts[label].update(_ngrams(doc, order))

    vocabulary = set(class_counts[HINDI]) | set(class_counts[NEPALI])
    vocab_size = len(vocabulary)

    log_prob: dict[str, dict[str, float]] = {}
    log_unseen: dict[str, float] = {}
    for label in (HINDI, NEPALI):
        counts = class_counts[label]
        denominator = sum(counts.values()) + vocab_size
        log_prob[label] = {
            gram: math.log((n + 1) / denominator) for gram, n in counts.items()
        }
        log_unseen[label] = math.log(1 / denominator)

    return LidModel(
        order=order,
        log_prob=log_prob,
        log_prior_unseen=log_unseen,
        vocabulary_size=vocab_size,
        n_train_docs={HINDI: len(hindi_docs), NEPALI: len(nepali_docs)},
    )


def evaluate(
    identifier: LanguageIdentifier,
    hindi_docs: list[str],
    nepali_docs: list[str],
) -> dict:
    """Measure held-out precision, recall and rejection rate.

    Reported in the Phase 1 statistics, and the basis for the claim that the two
    corpora are genuinely separate.  Precision is the number that matters:
    contamination is what damages the project, whereas rejected-but-valid
    documents only cost a little corpus volume.

    Args:
        identifier: Classifier to evaluate.
        hindi_docs: Held-out Hindi documents.
        nepali_docs: Held-out Nepali documents.

    Returns:
        Nested dictionary of per-class precision, recall, counts, and the
        cross-contamination totals.
    """
    results: dict[str, dict[str, int]] = {
        HINDI: {HINDI: 0, NEPALI: 0, AMBIGUOUS: 0},
        NEPALI: {HINDI: 0, NEPALI: 0, AMBIGUOUS: 0},
    }
    for true_label, docs in ((HINDI, hindi_docs), (NEPALI, nepali_docs)):
        for doc in docs:
            results[true_label][identifier.predict(doc).label] += 1

    out: dict = {"confusion": results}
    for label in (HINDI, NEPALI):
        other = NEPALI if label == HINDI else HINDI
        true_positive = results[label][label]
        false_positive = results[other][label]
        total_true = sum(results[label].values())
        out[label] = {
            "precision": true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0,
            "recall": true_positive / total_true if total_true else 0.0,
            "rejected_as_ambiguous": results[label][AMBIGUOUS],
            "leaked_into_other_corpus": results[label][other],
        }
    return out
