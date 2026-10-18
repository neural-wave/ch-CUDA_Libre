"""Statistical NER (spaCy via Presidio) plus Presidio's pattern recognizers for contact data and IDs.

Lower-cased ASR dialogues defeat cased NER models, so before running NER on such a text we build a
*truecased* copy: words that the (properly cased) note of the same encounter capitalizes mid-sentence
are capitalized in the dialogue too, as are turn starts and the pronoun "i". Only letter case changes,
so the copy has exactly the same length and every detected offset maps back 1:1.
"""

from __future__ import annotations

import re

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from clinanon import resources
from clinanon.detectors.rules import is_cased
from clinanon.types import Span

# Presidio entity -> taxonomy category. Statistical (spaCy) entities are weak evidence and go through
# the clinical veto; pattern-recognizer hits (e-mail, IBAN, ...) are reliable.
ENTITY_MAP = {
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "NRP": "DEMOGRAPHIC",
    "ORGANIZATION": "ORGANIZATION",
    "EMAIL_ADDRESS": "CONTACT",
    "PHONE_NUMBER": "CONTACT",
    "URL": "CONTACT",
    "IP_ADDRESS": "CONTACT",
    "US_SSN": "IDENTIFIER",
    "CREDIT_CARD": "IDENTIFIER",
    "IBAN_CODE": "IDENTIFIER",
}
_STATISTICAL = {"PERSON", "LOCATION", "NRP", "ORGANIZATION"}
# Pattern recognizers whose hits are ambiguous in clinical text ("169/74" looks like a phone number,
# "andrew.campbell" like a URL): they stay weak so the veto can still drop them.
_WEAK_PATTERNS = {"PHONE_NUMBER", "URL"}

# spaCy labels that are not personal information. ORG is deliberately *not* ignored (employers,
# schools, teams); its many false positives on drug names are removed by the clinical veto.
_SPACY_LABELS_TO_IGNORE = ["CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY", "ORDINAL", "PERCENT",
                           "PRODUCT", "QUANTITY", "WORK_OF_ART", "DATE", "TIME"]


def truecase(text: str, reference: str | None) -> str:
    """Same-length copy of a lower-cased text with likely proper nouns and sentence starts capitalized."""
    proper: set[str] = set()
    if reference:
        # Words the note always capitalizes (never writes in lower case) are proper nouns or drug
        # names; "The"/"Hypertension" also occur in lower case and drop out. Drugs are excluded too.
        capitalized = {w.lower() for w in re.findall(r"\b[A-Z][a-z'\-]+\b", reference)}
        lowercase = set(re.findall(r"\b[a-z][a-z'\-]+\b", reference))
        proper = capitalized - lowercase - resources.clinical_vocab() - resources.common_words()

    def cap(match: re.Match) -> str:
        word = match.group(0)
        upper = word[0].upper()
        return upper + word[1:] if len(upper) == 1 else word  # never change the length ("ß" -> "SS")

    out = re.sub(r"\b[^\W\d_][\w'\-]*", lambda m: cap(m) if m.group(0).lower() in proper else m.group(0), text)
    out = re.sub(r"\bi\b", "I", out)
    # Capitalize the first word of every turn and sentence: that is the distribution NER was trained on.
    out = re.sub(r"(^\[[^\]]+\]\s+|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out,
                 flags=re.M)
    assert len(out) == len(text)
    return out


class PresidioNer:
    def __init__(self, model: str = "en_core_web_lg", use_truecasing: bool = True):
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
            "ner_model_configuration": {"labels_to_ignore": _SPACY_LABELS_TO_IGNORE},
        })
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        self.use_truecasing = use_truecasing

    def detect(self, text: str, field: str, reference: str | None = None) -> list[Span]:
        """NER spans for one field; ``reference`` is the cased note used to truecase a dialogue."""
        analyzed = text
        if self.use_truecasing and not is_cased(text):
            analyzed = truecase(text, reference)
        spans = []
        for r in self.engine.analyze(text=analyzed, language="en", entities=list(ENTITY_MAP)):
            surface = text[r.start : r.end]
            if not surface.strip():
                continue
            statistical = r.entity_type in _STATISTICAL
            spans.append(Span(
                field=field, start=r.start, end=r.end, text=surface, category=ENTITY_MAP[r.entity_type],
                source=f"ner:{r.entity_type.lower()}" if statistical else f"pattern:{r.entity_type.lower()}",
                score=r.score, strong=not statistical and r.entity_type not in _WEAK_PATTERNS,
            ))
        return spans
