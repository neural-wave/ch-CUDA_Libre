"""Core data types shared by detectors, the clinical veto, the policy and the evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Free-text fields that are anonymized. Every other field (encounter_id, dataset, icd10_codes) is
# copied through untouched - the ICD-10 codes must be preserved by the task rules.
TEXT_FIELDS = ("note", "dialogue")


@dataclass
class Span:
    """A candidate piece of privacy-related information found by a detector."""

    field: str  # "note" or "dialogue"
    start: int  # character offsets into the field text
    end: int
    text: str
    category: str  # taxonomy category id, e.g. "NAME", "DATE" (see configs/privacy_taxonomy.yaml)
    source: str  # detector that produced it, e.g. "rule:title_name", "ner:spacy", "propagation"
    score: float = 1.0
    # Strong evidence (a title before a name, a dictation header, ...). Strong spans are never vetoed:
    # "mr. parkinson" is a person even though "parkinson" is also a disease.
    strong: bool = False
    role: str | None = None  # NAME only: "patient", "clinician" or None (unknown/other)

    def overlaps(self, other: Span) -> bool:
        return self.field == other.field and self.start < other.end and other.start < self.end


@dataclass
class Decision:
    """One anonymization decision, written to decisions.jsonl (debugging now, explainability later)."""

    encounter_id: str
    field: str
    start: int
    end: int
    text: str
    category: str
    action: str  # "tag" (replaced), "keep" (privacy category kept by policy) or "veto" (judged clinical)
    replacement: str | None
    source: str
    reason: str
    explanation: str = ""  # one readable sentence (clinanon.explain)
    confidence: float | None = None  # statistical detectors only

    def to_dict(self) -> dict:
        return asdict(self)
