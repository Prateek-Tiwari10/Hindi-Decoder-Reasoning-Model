"""Document quality filters for the cleaning pipeline.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Applied after Unicode normalization and before deduplication.  Each filter
returns a named reason on rejection rather than a bare boolean, because the
Phase 1 statistics report has to show how many documents were removed at each
stage -- and because a filter that silently eats 40% of a corpus is something
you want to see in a table, not discover after training.

The filters target what a news crawl actually produces after boilerplate
removal:

* **Navigation residue** -- menus and link lists that trafilatura did not strip,
  recognizable by very short lines with almost no sentence punctuation.
* **Template repetition** -- syndicated wire copy and standing disclaimers, which
  inflate a corpus without adding information and survive exact deduplication
  because a single differing word breaks the hash.
* **Wrong-script content** -- English and *romanized* Hindi, the latter
  being invisible to word-level filters but obvious at the character level.
* **Degenerate text** -- keyword-stuffed SEO pages and tables of figures, which
  a character-level model will happily memorize.

Thresholds are deliberately conservative.  Both corpora have far more raw text
available than the 500M-token target needs, so discarding a borderline-good
document costs nothing while keeping a bad one costs model quality.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from data.pipeline.normalize import devanagari_ratio

#: Sentence-terminating punctuation in Devanagari prose. Danda is the native
#: terminator; Latin marks appear in modern news text.
_SENTENCE_END = re.compile(r"[।॥.!?]")

_WORD_RE = re.compile(r"[ऀ-ॿ]+")

#: Lines that are almost certainly interface chrome rather than prose.
_BOILERPLATE_PATTERNS = re.compile(
    r"(?i)(cookie|javascript|advertisement|subscribe|newsletter|"
    r"copyright|all rights reserved|©|share this|read more|click here|"
    r"होम\s*पेज|शेयर करें|यहाँ क्लिक|सदस्यता|विज्ञापन)"
)


@dataclass
class QualityConfig:
    """Thresholds for :class:`QualityFilter`.

    Attributes:
        min_chars: Minimum document length. Below this a document carries too
            little context to be a useful training example.
        max_chars: Maximum length before truncation. Guards against a single
            scraped archive page dominating a shard.
        min_words: Minimum Devanagari word count.
        min_devanagari_ratio: Minimum fraction of non-space characters in
            Devanagari. The main defence against English and romanized text.
        max_digit_ratio: Maximum fraction of digit characters. Filters results
            tables, scoreboards and statistical listings.
        max_punctuation_ratio: Maximum fraction of punctuation characters.
        min_mean_line_length: Minimum mean characters per line. Navigation
            residue is many very short lines.
        max_repeated_line_ratio: Maximum fraction of lines that are duplicates
            within the document.
        max_top_bigram_ratio: Maximum share of any single word bigram. Catches
            keyword stuffing and templated text.
        max_boilerplate_line_ratio: Maximum fraction of lines matching known
            boilerplate patterns.
        min_sentence_punctuation_per_100w: Minimum sentence terminators per 100
            words. Running prose punctuates; link lists do not.
    """

    min_chars: int = 250
    max_chars: int = 200_000
    min_words: int = 40
    min_devanagari_ratio: float = 0.65
    max_digit_ratio: float = 0.15
    max_punctuation_ratio: float = 0.20
    min_mean_line_length: float = 25.0
    max_repeated_line_ratio: float = 0.30
    max_top_bigram_ratio: float = 0.05
    max_boilerplate_line_ratio: float = 0.25
    min_sentence_punctuation_per_100w: float = 1.5


@dataclass
class QualityStats:
    """Counts of documents accepted and rejected, by reason.

    Attributes:
        accepted: Documents that passed every filter.
        rejected: Rejection counts keyed by reason.
        chars_in: Characters seen before filtering.
        chars_out: Characters retained.
    """

    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    chars_in: int = 0
    chars_out: int = 0

    @property
    def total(self) -> int:
        """Total documents examined."""
        return self.accepted + sum(self.rejected.values())

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary for the statistics report.

        Returns:
            Totals, per-reason rejection counts, and retention ratios.
        """
        total = self.total
        return {
            "documents_in": total,
            "documents_accepted": self.accepted,
            "documents_rejected": sum(self.rejected.values()),
            "acceptance_rate": round(self.accepted / total, 4) if total else 0.0,
            "characters_in": self.chars_in,
            "characters_out": self.chars_out,
            "character_retention": (
                round(self.chars_out / self.chars_in, 4) if self.chars_in else 0.0
            ),
            "rejections_by_reason": dict(self.rejected.most_common()),
        }


class QualityFilter:
    """Accept or reject documents, recording why.

    Args:
        config: Threshold configuration.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        """Store thresholds and initialize statistics."""
        self.config = config or QualityConfig()
        self.stats = QualityStats()

    def check(self, text: str) -> str | None:
        """Test one document against every filter.

        Filters run cheapest-first so that the common rejections (too short,
        wrong script) are decided before the expensive n-gram statistics.

        Args:
            text: Normalized document text.

        Returns:
            ``None`` if the document passes, otherwise a short reason string
            naming the filter that rejected it.
        """
        config = self.config

        if len(text) < config.min_chars:
            return "too_short"
        if not text.strip():
            return "empty"

        ratio = devanagari_ratio(text)
        if ratio < config.min_devanagari_ratio:
            # Catches English pages and romanized Hindi alike; the latter
            # is common in comment sections and entertainment coverage.
            return "low_devanagari_ratio"

        words = _WORD_RE.findall(text)
        if len(words) < config.min_words:
            return "too_few_words"

        non_space = [c for c in text if not c.isspace()]
        if non_space:
            digits = sum(c.isdigit() for c in non_space)
            if digits / len(non_space) > config.max_digit_ratio:
                return "digit_heavy"
            # Punctuation must be counted by Unicode category, NOT by
            # `not c.isalnum()`. Devanagari vowel signs and the virama are
            # combining marks (categories Mn/Mc), for which Python's isalnum()
            # returns False -- so the naive test scores ordinary Devanagari prose
            # at ~0.37 "punctuation" and rejects nearly every document.
            # Matras are part of the letter they attach to, so only real
            # punctuation (P*) and symbols (S*) count here.
            punctuation = sum(
                unicodedata.category(c)[0] in ("P", "S") for c in non_space
            )
            if punctuation / len(non_space) > config.max_punctuation_ratio:
                return "punctuation_heavy"

        lines = [ln for ln in text.split("\n") if ln.strip()]
        if lines:
            mean_length = sum(len(ln) for ln in lines) / len(lines)
            if mean_length < config.min_mean_line_length:
                return "short_lines"

            unique = len(set(lines))
            if 1 - unique / len(lines) > config.max_repeated_line_ratio:
                return "repeated_lines"

            boilerplate = sum(bool(_BOILERPLATE_PATTERNS.search(ln)) for ln in lines)
            if boilerplate / len(lines) > config.max_boilerplate_line_ratio:
                return "boilerplate"

        sentence_marks = len(_SENTENCE_END.findall(text))
        if sentence_marks / len(words) * 100 < config.min_sentence_punctuation_per_100w:
            # Running prose punctuates regularly; link lists and tag clouds do
            # not, and they survive the line-length check when comma-separated.
            return "unpunctuated"

        if len(words) >= 20:
            bigrams = Counter(zip(words, words[1:]))
            top, count = bigrams.most_common(1)[0]
            if count / max(len(words) - 1, 1) > config.max_top_bigram_ratio:
                return "repetitive_bigram"

        return None

    def filter(self, text: str) -> str | None:
        """Check a document and update statistics.

        Args:
            text: Normalized document text.

        Returns:
            The retained text (truncated to ``max_chars``), or ``None`` if the
            document was rejected.
        """
        self.stats.chars_in += len(text)
        reason = self.check(text)
        if reason is not None:
            self.stats.rejected[reason] += 1
            return None
        kept = text[: self.config.max_chars]
        self.stats.accepted += 1
        self.stats.chars_out += len(kept)
        return kept
