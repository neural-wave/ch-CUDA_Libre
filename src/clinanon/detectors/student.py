"""The distilled student: a small token classifier fine-tuned on teacher-labelled data (scripts/train_student.py).

It recognises names, places, organizations and jobs from context, also in lower-cased ASR text and for
names missing from our (US) gazetteers. It runs on CPU (BERT-base, 110M parameters). Its spans are weak
evidence: the clinical veto still sees them.

Only the context-dependent categories are used. Ages, dates, times, IDs and contacts are formats that the
rules cover precisely; on real encounters the student read lab values as ages ("your alt was about 128").
Spans are extended to whole words, so "farmer" (farmer's lung) reaches the veto as a word, not as "farm".
"""

from __future__ import annotations

import re
from pathlib import Path

from clinanon.evaluation.judges import chunks
from clinanon.types import Span

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "student"
# Used when models/student/ is missing: downloaded once into the Hugging Face cache, pinned to a revision.
HUB_MODEL = "Arsxya/clinanon-student"
HUB_REVISION = "0ceeac40131c6cfe9c9b2bbcdbf3d3418c308977"
CONTEXT_CATEGORIES = frozenset({"NAME", "LOCATION", "ORGANIZATION", "PROFESSION"})
# Interjections the student reads as names in speech ("haha , i know", "god damn").
_NOT_NAMES = {"haha", "hah", "hehe", "damn", "dang", "darn", "gosh", "geez", "jeez", "golly", "heck", "ya", "yah",
              "yep", "yup", "nope", "okay", "ok", "huh", "wow", "ugh", "oops", "yay", "lol", "omg", "hmm", "mmm"}
_WORD_CHAR = re.compile(r"[\w'’\-]")


class StudentNer:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL, threshold: float = 0.75, device: str = "cpu",
                 categories: frozenset[str] = CONTEXT_CATEGORIES, revision: str | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), revision=revision)
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir), revision=revision)
        self.model = self.model.to(device).eval()
        self.labels = self.model.config.id2label
        self.threshold, self.device, self.categories = threshold, device, categories

    def detect(self, text: str, field: str, reference: str | None = None) -> list[Span]:
        spans: list[Span] = []
        pieces = [(o, c) for o, c in chunks(text, max_words=180) if c.strip()]
        if not pieces:
            return spans
        enc = self.tokenizer([c for _, c in pieces], truncation=True, max_length=512, padding=True,
                             return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        with self.torch.inference_mode():
            probs = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.softmax(-1).cpu()
        for (offset, _chunk), p, offs in zip(pieces, probs, offsets, strict=True):
            current = None  # [start, end, category, scores]
            for (a, b), token_probs in zip(offs.tolist(), p, strict=True):
                if a == b:
                    continue
                score, index = token_probs.max(-1)
                label = self.labels[int(index)]
                if label == "O":
                    current = self._close(current, spans, text, field)
                    continue
                prefix, category = label.split("-", 1)
                gap = text[current[1] : offset + a] if current else ""
                # continue a span over a sub-word split or an I- tag, but never across a line break
                if current and category == current[2] and "\n" not in gap and (prefix == "I" or not gap.strip()):
                    current[1] = offset + b  # continue the span (also over a sub-word split by the tokenizer)
                    current[3].append(float(score))
                else:
                    current = self._close(current, spans, text, field)
                    current = [offset + a, offset + b, category, [float(score)]]
            self._close(current, spans, text, field)
        return [s for s in spans if self._keep(s)]

    def _keep(self, span: Span) -> bool:
        if span.category not in self.categories or span.score < self.threshold:
            return False
        if span.category == "NAME":
            low = span.text.lower()
            return len(low) >= 3 and low not in _NOT_NAMES
        return True

    @staticmethod
    def _close(current, spans: list[Span], text: str, field: str):
        if current:
            start, end, category, scores = current
            while start > 0 and _WORD_CHAR.match(text[start - 1]):
                start -= 1  # whole words: "farm" inside "farmer" becomes "farmer"
            while end < len(text) and _WORD_CHAR.match(text[end]):
                end += 1
            surface = text[start:end]
            if surface.strip():
                spans.append(Span(field=field, start=start, end=end, text=surface, category=category,
                                  source=f"ner:student-{category.lower()}", score=sum(scores) / len(scores)))
        return None
