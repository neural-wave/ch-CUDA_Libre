"""Optional zero-shot NER with a GLiNER PII model (needs the ``judges`` dependency group).

GLiNER is prompted with label names, so it also recognises names, places and organizations that are
missing from our gazetteers (European names, Swiss insurers, foreign hospitals). Like spaCy, it runs on
the truecased copy of a lower-cased dialogue, and its hits are weak evidence that the clinical veto can
overrule. It runs on CPU or a small GPU (the base models have about 200-500M parameters).
"""

from __future__ import annotations

from clinanon.detectors.ner import truecase
from clinanon.detectors.rules import is_cased
from clinanon.evaluation.judges import _device, chunks
from clinanon.types import Span

DEFAULT_MODEL = "nvidia/gliner-PII"
# GLiNER label -> taxonomy category. Dates and ages are left to the rules (relative time must be kept).
LABELS = {
    "person": "NAME",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "address": "LOCATION",
    "phone number": "CONTACT",
    "email": "CONTACT",
    "medical record number": "IDENTIFIER",
    "health insurance number": "IDENTIFIER",
    "social security number": "IDENTIFIER",
    "id number": "IDENTIFIER",
}


class GlinerNer:
    def __init__(self, model_id: str = DEFAULT_MODEL, threshold: float = 0.5, use_truecasing: bool = True):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_id).to(_device())
        self.model_id, self.threshold, self.use_truecasing = model_id, threshold, use_truecasing

    def detect(self, text: str, field: str, reference: str | None = None) -> list[Span]:
        analyzed = text
        if self.use_truecasing and not is_cased(text):
            analyzed = truecase(text, reference)
        spans = []
        for offset, chunk in chunks(analyzed, max_words=150):
            for e in self.model.predict_entities(chunk, list(LABELS), threshold=self.threshold):
                start, end = offset + e["start"], offset + e["end"]
                surface = text[start:end]
                if not surface.strip():
                    continue
                spans.append(Span(field=field, start=start, end=end, text=surface, category=LABELS[e["label"]],
                                  source=f"ner:gliner-{e['label'].replace(' ', '_')}", score=e["score"]))
        return spans
