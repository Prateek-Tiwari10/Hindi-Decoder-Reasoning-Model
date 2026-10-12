"""Unicode normalization for Hindi Devanagari text.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Devanagari on the web is inconsistently encoded in ways that silently inflate a
BPE vocabulary: the same visible word can be several different codepoint
sequences.  Left unnormalized, the tokenizer spends vocabulary slots learning
that ``क़`` and ``क`` + ``़`` are the same word, and fertility (chars/token)
gets worse for no linguistic reason.

The main sources of inconsistency this module removes:

* **Nukta forms.**  ``क़`` exists both precomposed (U+0958) and decomposed
  (U+0915 U+093C).  The precomposed characters are on the Unicode composition
  exclusion list, so NFC canonically maps them to the decomposed form -- which
  is what we want, and why NFC (not NFD/NFKC) is the right choice here.
* **Zero-width joiners.**  ZWJ/ZWNJ control conjunct rendering.  They are
  invisible, wildly inconsistent across sources, and would otherwise become
  tokenizer noise.
* **Digit systems.**  Devanagari digits (०-९) and ASCII digits are freely mixed
  within the same corpus; unifying them halves the digit vocabulary.
* **Punctuation variants.**  Curly quotes, en/em dashes, ellipsis characters and
  non-breaking spaces arrive from HTML in many forms.

Design note: the behaviour is driven by a :class:`NormalizerConfig`.  The Hindi
corpus was built with chandrabindu folding ON (``ँ`` -> ``ं``); see
:attr:`NormalizerConfig.fold_chandrabindu`.

This module is pure text -> text.  It holds no corpus, no vocabulary and no
state, so sharing it between the two language pipelines does not violate the
project's model-independence constraint.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- Character class constants ---------------------------------------------

#: The Devanagari Unicode block (U+0900-U+097F).
DEVANAGARI_RANGE = (0x0900, 0x097F)

#: Devanagari Extended and Vedic Extensions -- rare, mostly Vedic accents that
#: carry no value for a news/prose corpus and are stripped as noise.
DEVANAGARI_EXTENDED_RANGE = (0xA8E0, 0xA8FF)
VEDIC_EXTENSIONS_RANGE = (0x1CD0, 0x1CFF)

#: Danda and double danda: Devanagari sentence terminators.  Kept, and treated
#: as sentence boundaries downstream.
DANDA = "।"
DOUBLE_DANDA = "॥"

#: Devanagari digits (U+0966-U+096F) -> ASCII.
_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_DIGIT_TABLE = {ord(d): str(i) for i, d in enumerate(_DEVANAGARI_DIGITS)}

#: Invisible characters that must never reach the tokenizer.  ZWJ/ZWNJ are
#: handled separately because their removal is configurable.
_INVISIBLE = {
    0x00AD,  # soft hyphen
    0x200B,  # zero-width space
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x2060,  # word joiner
    0xFEFF,  # zero-width no-break space / BOM
}
_INVISIBLE_TABLE = {cp: None for cp in _INVISIBLE}

_ZWNJ = "‌"
_ZWJ = "‍"

#: Punctuation and space variants that HTML extraction leaves behind.
_PUNCT_TABLE = {
    0x00A0: " ",   # non-breaking space
    0x2007: " ",   # figure space
    0x202F: " ",   # narrow no-break space
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2032: "'", 0x2033: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-",
    0x2014: "-", 0x2015: "-", 0x2212: "-",
    0x2026: "...",  # horizontal ellipsis
}

#: Chandrabindu -> anusvara.  Off by default; see the config field.
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"

# --- Precompiled patterns ---------------------------------------------------

_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_HSPACE = re.compile(r"[ \t\f\r]+")
_RE_BLANKLINES = re.compile(r"\n{3,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:!?%।॥])")
# Collapse runs of 4+ down to 3, so a normalized ellipsis ("...") survives
# intact while "!!!!!!!" is truncated.
_RE_REPEAT_PUNCT = re.compile(r"([!?.।॥])\1{3,}")
_RE_REPEAT_CHAR = re.compile(r"(\S)\1{3,}")
# Insert a space after a danda that is glued to the next word, but not when the
# next character is itself a danda (a repeated-danda run is noise, not a
# sentence boundary).
_RE_DANDA_SPACING = re.compile(r"([।॥])(?=[^\s।॥])")


@dataclass(frozen=True)
class NormalizerConfig:
    """Per-language switches for :class:`DevanagariNormalizer`.

    Attributes:
        strip_zero_width: Remove ZWJ/ZWNJ.  These only affect conjunct
            rendering, never meaning, and are applied inconsistently across
            sources, so removing them is a net win for tokenizer consistency.
        devanagari_digits_to_ascii: Map ०-९ to 0-9.  Corpora mix both systems
            freely; unifying them prevents the tokenizer from learning two
            parallel sets of number tokens.
        fold_chandrabindu: Map chandrabindu (ँ) to anusvara (ं).

            **Hindi:** the two are frequently interchanged in informal web text,
            so folding reduces spurious variation.

        collapse_repeated_chars: Truncate runs of 4+ identical non-space
            characters (``"क्याााााा"``) down to three.  Targets social/comment
            text that survives boilerplate removal.
        strip_vedic_marks: Remove Vedic accent marks and Devanagari Extended
            characters, which are noise outside liturgical text.
    """

    strip_zero_width: bool = True
    devanagari_digits_to_ascii: bool = True
    fold_chandrabindu: bool = False
    collapse_repeated_chars: bool = True
    strip_vedic_marks: bool = True

    @classmethod
    def for_hindi(cls) -> "NormalizerConfig":
        """Return the Hindi configuration (chandrabindu folding enabled)."""
        return cls(fold_chandrabindu=True)

    @classmethod
    def for_nepali(cls) -> "NormalizerConfig":
        """Return the Nepali configuration -- used ONLY by ``train_lid.py``.

        Training the filter that keeps only Hindi documents needs Nepali
        negatives normalised the way Nepali text is written: chandrabindu
        folding off, because nasalization is phonemic in Nepali.
        """
        return cls(fold_chandrabindu=False)


class DevanagariNormalizer:
    """Normalize Devanagari text to a single canonical encoding.

    The transformation is deterministic and idempotent: normalizing an already
    normalized string returns it unchanged.  That property is relied upon by the
    deduplication stage, which hashes normalized text.

    Example:
        >>> norm = DevanagariNormalizer(NormalizerConfig.for_hindi())
        >>> norm("क़‍र  १२३ ।")
        'क़र 123।'
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        """Store the configuration and precompute the translation table.

        Args:
            config: Language-specific switches.  Defaults to
                :class:`NormalizerConfig` defaults (safe for either language,
                with chandrabindu folding off).
        """
        self.config = config or NormalizerConfig()
        self._table = self._build_translation_table()

    def _build_translation_table(self) -> dict[int, str | None]:
        """Build the single ``str.translate`` table for all codepoint mappings.

        Combining every codepoint-level substitution into one table means the
        string is walked once rather than once per rule, which matters when
        normalizing on the order of 10^9 characters.

        Returns:
            Mapping from codepoint to replacement string, or to ``None`` for
            deletion.
        """
        table: dict[int, str | None] = {}
        table.update(_INVISIBLE_TABLE)
        table.update(_PUNCT_TABLE)

        if self.config.strip_zero_width:
            table[ord(_ZWNJ)] = None
            table[ord(_ZWJ)] = None

        if self.config.devanagari_digits_to_ascii:
            table.update(_DIGIT_TABLE)

        if self.config.fold_chandrabindu:
            table[ord(_CHANDRABINDU)] = _ANUSVARA

        if self.config.strip_vedic_marks:
            for lo, hi in (VEDIC_EXTENSIONS_RANGE, DEVANAGARI_EXTENDED_RANGE):
                for cp in range(lo, hi + 1):
                    table[cp] = None

        return table

    def __call__(self, text: str) -> str:
        """Alias for :meth:`normalize` so instances can be used as functions."""
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """Normalize a document or line of Devanagari text.

        Applies, in order: NFC composition, codepoint translation, control
        character removal, repetition collapsing, and whitespace tidying.  The
        order matters -- NFC runs first so that the translation table sees
        canonical codepoints, and whitespace tidying runs last so it cleans up
        gaps left by deleted characters.

        Args:
            text: Raw text, typically one extracted document.

        Returns:
            Normalized text.  Returns ``""`` for empty or whitespace-only input.
        """
        if not text:
            return ""

        # NFC first: canonically decomposes the excluded precomposed nukta
        # characters and composes everything else, so downstream rules see one
        # consistent form.
        text = unicodedata.normalize("NFC", text)
        text = text.translate(self._table)
        text = _RE_CONTROL.sub("", text)

        if self.config.collapse_repeated_chars:
            text = _RE_REPEAT_CHAR.sub(r"\1\1\1", text)
        text = _RE_REPEAT_PUNCT.sub(r"\1\1\1", text)

        return self._normalize_whitespace(text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse horizontal whitespace and tidy punctuation spacing.

        Newlines are preserved (capped at two consecutive) because paragraph
        structure is a useful signal for the quality filters and for sentence
        splitting; only horizontal whitespace is collapsed.

        Args:
            text: Text after codepoint-level normalization.

        Returns:
            Text with tidy spacing and no leading/trailing whitespace.
        """
        text = _RE_HSPACE.sub(" ", text)
        text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
        # A danda immediately followed by a word is a missing-space artefact of
        # HTML extraction; insert the space so sentence splitting works.
        text = _RE_DANDA_SPACING.sub(r"\1 ", text)
        text = _RE_BLANKLINES.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()


def devanagari_ratio(text: str) -> float:
    """Fraction of non-space characters that are Devanagari.

    This is the primary script filter: Hindi documents sit well above
    0.6, while English pages, transliterated (romanized) Hindi, and
    navigation-chrome fragments fall far below it.  Romanized Hindi is a real
    contaminant on Indian news sites and is invisible to word-level filters,
    which is why the check is at the character level.

    Args:
        text: Text to measure, normally already normalized.

    Returns:
        Ratio in ``[0.0, 1.0]``.  Returns ``0.0`` when the text has no
        non-space characters.
    """
    lo, hi = DEVANAGARI_RANGE
    total = 0
    devanagari = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if lo <= ord(ch) <= hi:
            devanagari += 1
    return devanagari / total if total else 0.0


def count_characters(text: str) -> tuple[int, int]:
    """Count characters and UTF-8 bytes, for corpus statistics.

    Both numbers are required by the Phase 1 statistics report, and their ratio
    is the empirical ``bytes/char`` constant that the corpus sizing math depends
    on (predicted ~2.68 for cleaned Devanagari, since Devanagari codepoints cost
    3 UTF-8 bytes and spaces/ASCII cost 1).

    Args:
        text: Text to measure.

    Returns:
        Tuple of ``(character_count, utf8_byte_count)``.
    """
    return len(text), len(text.encode("utf-8"))
