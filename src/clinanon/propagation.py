"""Record-level propagation: a name found anywhere in an encounter is removed everywhere in it.

The cased note and the dictation header reveal names reliably ("Andrew Campbell is a 59-year-old"),
while the lower-cased dialogue mentions the same people without any cue ("so , andrew , what's going
on ?"). Propagating every accepted name token across both fields closes that gap.

Stop words ("will") are never propagated. Other names that are ordinary words ("mark", "brown") are
propagated only when a strong rule identified them as a name, and the veto keeps mentions that are used
as words ("a mark on the skin", "dark brown stool").

Places and organizations propagate the same way, as whole multi-word names: "Point May Orthopedics" in
the note removes "point may orthopedics" in the dialogue, and a spoken short form of a long word counts
too ("East Metropolitan Hospital" -> "east metro").
"""

from __future__ import annotations

import re

from spacy.lang.en.stop_words import STOP_WORDS

from clinanon import resources
from clinanon.detectors.rules import NON_NAME_WORDS, is_cased, is_frequent_name
from clinanon.types import Span
from clinanon.veto import RELATIONSHIP_WORDS, is_known_name

_NAME_PART = re.compile(r"[^\W\d_][^\W\d_'’\-]*(?:['’\-][^\W\d_]+)*")
_ROLE_PRIORITY = {"clinician": 2, "patient": 1, None: 0}


def _can_seed(span: Span, texts: dict[str, str]) -> bool:
    """Which accepted NAME spans may spread their tokens across the record.

    Rules (strong) always may; NER in a cased note is reliable enough; NER in a lower-cased dialogue
    only when every token is a known name - otherwise one NER mistake ("wan na") is copied everywhere.
    """
    if span.strong or is_cased(texts[span.field]):
        return True
    return all(is_known_name(t) for t in _NAME_PART.findall(span.text.lower()))


def collect_names(spans: list[Span], texts: dict[str, str]) -> dict[str, str | None]:
    """Map name token (lower case) -> role, from accepted NAME spans that may seed propagation."""
    names: dict[str, str | None] = {}
    for span in spans:
        if span.category != "NAME" or not _can_seed(span, texts):
            continue
        for part in _NAME_PART.findall(span.text):
            token = part.lower()
            if len(token) < 2 or token in STOP_WORDS or token in RELATIONSHIP_WORDS or token in NON_NAME_WORDS:
                continue
            is_word = token in resources.common_words()
            is_clinical = token in resources.clinical_vocab()
            # Names that are also ordinary words ("mark") spread only from strong evidence, only if they
            # are frequent names, and never if the word is also clinical ("walker", "sharp", "long").
            # Their mentions are re-checked by the veto ("a mark", "your walker" stay words).
            if is_word and not (span.strong and is_frequent_name(token) and not is_clinical):
                continue
            # Clinical eponyms ("thompson", "wilson") spread only from strong evidence; the veto keeps
            # the clinical mentions ("thompson test", "wilson's disease").
            if is_clinical and not span.strong:
                continue
            if token not in names or _ROLE_PRIORITY[span.role] > _ROLE_PRIORITY[names[token]]:
                names[token] = span.role
    return names


def find_mentions(texts: dict[str, str], names: dict[str, str | None]) -> list[Span]:
    """All case-insensitive, whole-word mentions of the collected names in every field."""
    spans = []
    for token, role in names.items():
        pattern = re.compile(rf"(?<![\w'’\-]){re.escape(token)}(?![\w\-])", re.I)
        for field, text in texts.items():
            for m in pattern.finditer(text):
                spans.append(Span(field, m.start(), m.end(), m.group(0), "NAME", "propagation", role=role))
    return spans


_PLACE_CATEGORIES = {"ORGANIZATION", "LOCATION"}
_SHORT_FORM_MIN = 4  # "metro" for "metropolitan": the spoken form keeps at least the first 4 letters


def collect_places(spans: list[Span], texts: dict[str, str]) -> dict[str, str]:
    """Map multi-word place/organization name (lower case) -> category, from reliable accepted spans:
    rule spans, or NER spans in a cased field. Single words are not propagated ("Johnson" is also a name).
    """
    places: dict[str, str] = {}
    for span in spans:
        if span.category not in _PLACE_CATEGORIES or not (span.strong or is_cased(texts[span.field])):
            continue
        tokens = _NAME_PART.findall(span.text.lower())
        if len(tokens) >= 2 and tokens[0] not in STOP_WORDS:
            places.setdefault(" ".join(tokens), span.category)
    return places


def find_place_mentions(texts: dict[str, str], places: dict[str, str]) -> list[Span]:
    """Mentions of the collected places in every field: the full name (any casing or spacing) and the
    short form "first word + start of the second word" when the second word is long."""
    spans = []
    for name, category in places.items():
        first, second, *rest = name.split()
        forms = [r"\s+".join(map(re.escape, name.split()))]
        if len(second) > _SHORT_FORM_MIN:
            forms.append(rf"{re.escape(first)}\s+{re.escape(second[:_SHORT_FORM_MIN])}[^\W\d_]*")
        pattern = re.compile(rf"(?<![\w'’\-])(?:{'|'.join(forms)})(?![\w\-])", re.I)
        for field, text in texts.items():
            for m in pattern.finditer(text):
                spans.append(Span(field, m.start(), m.end(), m.group(0), category, "propagation"))
    return spans
