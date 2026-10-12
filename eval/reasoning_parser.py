"""Deterministic parser for the emitted chain of thought.

WHY PARSE THE OUTPUT AT ALL
---------------------------
Exact match tells you whether the final label is right.  It cannot tell you
*why* it is wrong, and the two failure modes this project cares about are
distinguishable only inside the chain of thought:

* the model builds the wrong order and then answers it correctly;
* the model builds the RIGHT order and then contradicts it.

The second is the bound bug surfaced in Part 2, and it is invisible to EM
alone.  Separating them needs the emitted `क्रम:` chain read back out of the
generated text, which is what this module does.

LANGUAGE-PARAMETRIC, NOT LANGUAGE-SPECIFIC
------------------------------------------
Every connective comes from the Part 1 language config -- the same file the
generator wrote the completions with -- so the parser cannot drift away from
the data.  The two languages share one code path because their completions
share one structure, which is a property Part 1 deliberately built in by
generating both from the same seed and plan.

FAILING LOUDLY
--------------
A completion this parser cannot read is recorded as ``unparseable`` with its
record id, never silently scored as wrong.  The distinction matters: a wrong
answer is a fact about the model, an unparseable one may be a fact about the
parser, and conflating them would let a parser bug masquerade as a model
failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Danda.  Completion segments are joined with it by the Part 1 generator.
DANDA = "।"


@dataclass
class ParsedCompletion:
    """What could be read back out of one generated completion.

    Attributes:
        chain: Entities in the emitted ``क्रम:`` chain, greatest first.
        label: The final answer label under the G8 rule.
        bound_entity: Subject of the bound-propagation step, if present.
        bound_anchor: Entity the bound step compares against.
        bound_symbol: ``"<"`` or ``">"`` as emitted in the bound step.
        normalizations: The ``A > B`` pairs emitted in the normalization span.
        has_chain_marker: Whether a ``क्रम:`` marker was present at all.
        has_answer_cue: Whether a terminal answer cue was present.
        unparseable: Reasons the completion could not be fully read.
    """

    chain: list[str] = field(default_factory=list)
    label: str | None = None
    bound_entity: str | None = None
    bound_anchor: str | None = None
    bound_symbol: str | None = None
    normalizations: list[tuple[str, str]] = field(default_factory=list)
    has_chain_marker: bool = False
    has_answer_cue: bool = False
    unparseable: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if both the chain and the label were recovered."""
        return bool(self.chain) and self.label is not None


class CompletionParser:
    """Reads generated completions for one language."""

    def __init__(self, config: dict) -> None:
        """Build a parser from a Part 1 language config.

        Args:
            config: The parsed ``configs/<code>.yaml``.
        """
        self.answer_cue = config["cues"]["answer"]
        self.order_marker = config["connectives"]["order"]
        self.namely = config["connectives"]["namely"]
        self.then = config["connectives"]["then"]
        self.therefore = config["connectives"]["therefore"]
        self.affirmative = config["labels"]["affirmative"]
        self.negative = config["labels"]["negative"]

        # `तो` (hi) is a common word; `त्यसो भए` (ne) is not.  Anchoring the
        # bound step to a segment START rather than searching anywhere in the
        # text keeps the Hindi case from matching mid-sentence occurrences.
        self.bound_pattern = re.compile(
            re.escape(self.then) + r"\s+(\S+)\s+([<>])\s+(\S+)\s*=")

    def segments(self, text: str) -> list[str]:
        """Split a completion into its danda-terminated clauses.

        Args:
            text: The generated completion.

        Returns:
            Stripped, non-empty segments.
        """
        return [s.strip() for s in text.split(DANDA) if s.strip()]

    def parse_chain(self, text: str) -> tuple[list[str], bool]:
        """Read the ``क्रम:`` chain.

        The chain runs from the marker to the next danda, and its entities are
        separated by ``>``.  The FIRST marker is used: a degenerate generation
        can emit several, and the first is the one the rest of the completion
        was conditioned on.

        Args:
            text: The generated completion.

        Returns:
            ``(entities, marker_present)``.
        """
        position = text.find(self.order_marker)
        if position < 0:
            return [], False
        tail = text[position + len(self.order_marker):]
        clause = tail.split(DANDA, 1)[0]
        entities = [part.strip() for part in clause.split(">")]
        entities = [e for e in entities if e]
        return entities, True

    def parse_label(self, text: str) -> str | None:
        """Apply the G8 rule: text after the LAST answer cue, minus the danda.

        Args:
            text: The generated completion.

        Returns:
            The label, or None if no cue is present.
        """
        position = text.rfind(self.answer_cue)
        if position < 0:
            return None
        tail = text[position + len(self.answer_cue):].strip()
        if tail.endswith(DANDA):
            tail = tail[:-1].strip()
        return tail or None

    def parse_bound_step(self, text: str
                         ) -> tuple[str | None, str | None, str | None]:
        """Read the bound-propagation step of a numeric or mixed completion.

        Args:
            text: The generated completion.

        Returns:
            ``(entity, symbol, anchor)``, each None if absent.
        """
        for segment in self.segments(text):
            if not segment.startswith(self.then):
                continue
            match = self.bound_pattern.search(segment)
            if match:
                return match.group(1), match.group(2), match.group(3)
        return None, None, None

    def parse_normalizations(self, text: str) -> list[tuple[str, str]]:
        """Read the normalized ``A > B`` pairs from the normalization span.

        Only the span before the chain marker is considered, so the chain
        itself is not mistaken for a list of normalizations.

        Args:
            text: The generated completion.

        Returns:
            ``(greater, lesser)`` pairs in emitted order.
        """
        head = text.split(self.order_marker, 1)[0]
        pairs = []
        for clause in head.replace(DANDA, ",").split(","):
            clause = clause.strip()
            if not clause:
                continue
            # A flipped premise is restated as `B < A यानी A > B`; take the
            # part after the connective, which is already normalized.
            if self.namely in clause:
                clause = clause.split(self.namely, 1)[1].strip()
            if ">" in clause and "<" not in clause:
                parts = [p.strip() for p in clause.split(">")]
                if len(parts) == 2 and all(parts):
                    pairs.append((parts[0], parts[1]))
        return pairs

    def parse(self, text: str, record_id: str = "") -> ParsedCompletion:
        """Parse one completion.

        Args:
            text: The generated completion.
            record_id: For the unparseable log.

        Returns:
            The parse, with ``unparseable`` listing anything not recovered.
        """
        chain, has_marker = self.parse_chain(text)
        label = self.parse_label(text)
        entity, symbol, anchor = self.parse_bound_step(text)

        reasons = []
        if not has_marker:
            reasons.append("no chain marker")
        elif not chain:
            reasons.append("empty chain")
        if label is None:
            reasons.append("no answer cue")

        return ParsedCompletion(
            chain=chain, label=label, bound_entity=entity,
            bound_anchor=anchor, bound_symbol=symbol,
            normalizations=self.parse_normalizations(text),
            has_chain_marker=has_marker, has_answer_cue=label is not None,
            unparseable=reasons)
