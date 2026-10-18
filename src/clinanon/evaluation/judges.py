"""Stand-ins for the organizers' privacy detector ("judge").

The organizers score privacy by running a PII detector over our output, but we do not know which one.
We score under several plausible judges and calibrate them against the official mid-term number:

* ``presidio-default`` - Presidio out of the box (spaCy en_core_web_lg + predefined recognizers). The most
  common open-source choice. Very noisy on clinical text: flags "today", "daily", "C7", "Mobic", ...
* ``presidio-strict``  - the same detections, minus what a sensible organizer would filter out: relative
  time expressions, durations, frequencies and the noisy US_* number recognizers.
* ``gliner-*``         - zero-shot GLiNER PII models, prompted with a label list (with and without
  date/age labels, since we do not know whether the organizers count those).
* ``deid-*``           - transformer models fine-tuned for clinical de-identification (i2b2-style PHI).

The model-based judges need the optional ``judges`` dependency group (``uv sync --group judges``) and are
imported lazily, so the anonymizer itself never depends on them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from presidio_analyzer import AnalyzerEngine


@dataclass(frozen=True)
class Entity:
    label: str
    start: int
    end: int
    text: str


class Judge(Protocol):
    name: str

    def detect(self, text: str) -> list[Entity]: ...


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# DATE_TIME detections that refer to a calendar date or an age (kept by the strict judge).
_CALENDAR_OR_AGE = re.compile(
    rf"\b(?:19|20)\d{{2}}\b|\b\d{{1,2}}[/.-]\d{{1,2}}(?:[/.-]\d{{2,4}})?\b|\b{_MONTH}\b|year[- ]old|years old",
    re.IGNORECASE,
)
_NOISY_RECOGNIZER_LABELS = {"US_DRIVER_LICENSE", "US_BANK_NUMBER", "US_ITIN", "US_PASSPORT"}


@cache
def _presidio() -> AnalyzerEngine:
    return AnalyzerEngine()


class PresidioJudge:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.name = "presidio-strict" if strict else "presidio-default"

    def detect(self, text: str) -> list[Entity]:
        entities = []
        for r in _presidio().analyze(text=text, language="en"):
            span = text[r.start : r.end]
            if self.strict:
                if r.entity_type in _NOISY_RECOGNIZER_LABELS:
                    continue
                if r.entity_type == "DATE_TIME" and not _CALENDAR_OR_AGE.search(span):
                    continue
            entities.append(Entity(r.entity_type, r.start, r.end, span))
        return entities


# ------------------------------------------------------------------------------------- model-based judges


def chunks(text: str, max_words: int = 150) -> Iterator[tuple[int, str]]:
    """Split ``text`` into (offset, chunk) pieces of at most ``max_words`` words.

    Transformer judges only see a few hundred tokens at a time. We cut at line breaks (dialogue turns,
    note sections) and, inside long lines, after sentence ends, so that no entity is cut in half.
    """
    pieces: list[tuple[int, int]] = []  # (start, end) of units that must stay together
    for line in re.finditer(r"[^\n]+", text):
        start = line.start()
        for sentence_end in re.finditer(r"(?<=[.?!])\s+", line.group()):
            pieces.append((start, line.start() + sentence_end.end()))
            start = line.start() + sentence_end.end()
        pieces.append((start, line.end()))

    chunk_start, chunk_end, words = None, None, 0
    for start, end in pieces:
        n = len(text[start:end].split())
        if chunk_start is not None and words + n > max_words:
            yield chunk_start, text[chunk_start:chunk_end]
            chunk_start, words = None, 0
        if chunk_start is None:
            chunk_start = start
        chunk_end, words = end, words + n
    if chunk_start is not None:
        yield chunk_start, text[chunk_start:chunk_end]


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


# Label prompts for the zero-shot GLiNER judges: what an organizer would plausibly ask for.
PII_LABELS = [
    "person", "organization", "location", "address", "phone number", "email", "medical record number",
    "social security number", "health insurance number", "id number",
]  # fmt: skip
TIME_LABELS = ["date", "age"]


class GlinerJudge:
    def __init__(self, name: str, model_id: str, labels: list[str], threshold: float = 0.5):
        self.name, self.model_id, self.labels, self.threshold = name, model_id, labels, threshold

    def detect(self, text: str) -> list[Entity]:
        model = _gliner(self.model_id)
        entities = []
        for offset, chunk in chunks(text):
            for e in model.predict_entities(chunk, self.labels, threshold=self.threshold):
                start, end = offset + e["start"], offset + e["end"]
                entities.append(Entity(e["label"].upper(), start, end, text[start:end]))
        return entities


@cache
def _gliner(model_id: str):
    from gliner import GLiNER

    return GLiNER.from_pretrained(model_id).to(_device())


class TokenClassifierJudge:
    """A Hugging Face token-classification model whose every label is a privacy category."""

    def __init__(self, name: str, model_id: str):
        self.name, self.model_id = name, model_id

    def detect(self, text: str) -> list[Entity]:
        pipe = _token_pipeline(self.model_id)
        entities: list[Entity] = []
        for offset, chunk in chunks(text, max_words=120):
            for e in pipe(chunk):
                # Some models use BILOU tags ("U-AGE", "L-DATE") that the pipeline does not strip.
                label = re.sub(r"^[BILUES]-", "", e["entity_group"])
                start, end = offset + e["start"], offset + e["end"]
                prev = entities[-1] if entities else None
                # Re-join word pieces of one entity that the aggregation left split ("Mar" + "tha").
                if prev and prev.label == label and not text[prev.end : start].strip(" -"):
                    entities[-1] = Entity(label, prev.start, end, text[prev.start : end])
                else:
                    entities.append(Entity(label, start, end, text[start:end]))
        return [e for e in entities if e.text.strip()]


@cache
def _token_pipeline(model_id: str):
    from transformers import pipeline

    return pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=_device())


def default_judges() -> list[Judge]:
    """Judges run by ``clinanon evaluate`` (fast, no optional dependencies)."""
    return [PresidioJudge(strict=False), PresidioJudge(strict=True)]


def candidate_judges() -> list[Judge]:
    """All judges compared during calibration (needs the ``judges`` dependency group)."""
    judges: list[Judge] = default_judges()
    for short, model_id in [
        ("gliner-multi-pii", "urchade/gliner_multi_pii-v1"),
        ("gliner-nvidia-pii", "nvidia/gliner-PII"),
        ("gliner-knowledgator-pii", "knowledgator/gliner-pii-base-v1.0"),
    ]:
        judges.append(GlinerJudge(short, model_id, PII_LABELS))
        judges.append(GlinerJudge(f"{short}+date-age", model_id, PII_LABELS + TIME_LABELS))
    judges.append(TokenClassifierJudge("deid-obi-i2b2", "obi/deid_roberta_i2b2"))
    judges.append(TokenClassifierJudge("deid-stanford", "StanfordAIMI/stanford-deidentifier-base"))
    return judges
