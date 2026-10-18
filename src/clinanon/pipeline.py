"""The anonymization pipeline: detect -> veto -> propagate -> veto -> resolve overlaps -> replace.

    record ──► rules + NER on note and dialogue ──► clinical veto ──► accepted spans
                                                                       │
               names propagated across both fields ◄───────────────────┘
                     │
                     ▼
               clinical veto (context only) ──► overlap resolution ──► typed tags ──► record'

Only the detected spans change; every other character of the text is preserved (minimal-diff design),
and the icd10_codes field is copied untouched.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace as dc_replace

from clinanon import propagation, veto
from clinanon.detectors import rules
from clinanon.detectors.ner import PresidioNer
from clinanon.explain import evidence, explain, explain_veto
from clinanon.taxonomy import DEFAULT_TAXONOMY, Taxonomy
from clinanon.types import TEXT_FIELDS, Decision, Span

log = logging.getLogger("clinanon")

# Words of ICD descriptions that do not identify the diagnosis (excluded from the per-record protect list).
_ICD_GENERIC = {
    "unspecified", "without", "other", "specified", "initial", "encounter", "subsequent", "sequela",
    "elsewhere", "classified", "site", "part", "unknown", "type", "region", "complications", "sites",
}  # fmt: skip
# When candidate spans overlap, the most reliable evidence wins; ties go to the longer span.
_SOURCE_RANK = {"rule": 3, "pattern": 2, "propagation": 2, "ner": 1, "gazetteer": 1}
# Document structure that must never be edited, whatever a detector says: speaker tags at the start of
# a dialogue turn ("[patient]") and ALL-CAPS section headers of a note ("HISTORY OF PRESENT ILLNESS").
_STRUCTURE = re.compile(r"^\[[^\]\n]{1,30}\]|^[A-Z][A-Z0-9 &/,()\-]{2,}:?[ \t]*$", re.M)


def _in_structure(span: Span, texts: dict[str, str]) -> bool:
    return any(m.start() < span.end and span.start < m.end() for m in _STRUCTURE.finditer(texts[span.field]))


def _drop_structure(spans: list[Span], texts: dict[str, str]) -> tuple[list[Span], list[tuple[Span, str]]]:
    kept, dropped = [], []
    for span in spans:
        if _in_structure(span, texts):
            dropped.append((span, "document structure (speaker tag / section header)"))
        else:
            kept.append(span)
    return kept, dropped


def record_protect_terms(record: dict) -> set[str]:
    """Content words of this encounter's ICD-10 descriptions (available at inference by the task rules)."""
    words = set()
    for code in record.get("icd10_codes", []):
        for word in re.findall(r"[a-z][a-z'\-]+", code["description"].lower()):
            if len(word) >= 4 and word not in _ICD_GENERIC:
                words.add(word.removesuffix("'s"))
    return words


def _priority(span: Span) -> tuple:
    return (span.strong, _SOURCE_RANK[span.source.split(":")[0]], span.role is not None, span.end - span.start)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy selection of non-overlapping spans, best evidence first."""
    chosen: list[Span] = []
    for span in sorted(spans, key=_priority, reverse=True):
        if not any(span.overlaps(c) for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: (s.field, s.start))


# ASR transcripts split some contractions into two tokens ("wan na", "lem me", "gon na"); PII detectors
# read the first half as a person's name (GLiNER also flags "gon na"). Rejoining them changes no
# information. Only the forms that trigger detectors are touched ("do n't" is left as it is).
_ASR_SPLITS = [
    (re.compile(r"\bwan na\b", re.I), "wanna"),
    (re.compile(r"\blem me\b", re.I), "lemme"),
    (re.compile(r"\bgon na\b", re.I), "gonna"),
]


def normalize_asr_splits(text: str) -> str:
    for pattern, joined in _ASR_SPLITS:
        text = pattern.sub(joined, text)
    return text


def _load_student(student_model: str | None):
    """The distilled student: a directory, "auto" (models/student/, else the pinned Hugging Face model) or None."""
    if not student_model:
        return None
    from clinanon.detectors import student

    if student_model != "auto":
        source, revision = student_model, None
    elif (student.DEFAULT_MODEL / "config.json").exists():
        source, revision = student.DEFAULT_MODEL, None
    else:
        source, revision = student.HUB_MODEL, student.HUB_REVISION
    try:
        detector = student.StudentNer(source, revision=revision)
    except (ImportError, OSError) as error:  # torch missing, offline, or model not reachable
        log.warning("distilled student unavailable (%s): running rules + spaCy only", error)
        return None
    log.info("distilled student: %s", source)
    return detector


def _first_line(span: Span) -> Span | None:
    """A line break separates fields, turns or sentences, never the parts of one value: NER spans such as
    "Anna\\nDOB" are cut at the first line break ("DOB" would otherwise spread as a name)."""
    if "\n" not in span.text:
        return span
    head = span.text.split("\n", 1)[0].rstrip()
    return dc_replace(span, end=span.start + len(head), text=head) if head.strip() else None


def _confidence(span: Span) -> float | None:
    return round(span.score, 3) if span.source.startswith(("ner", "pattern")) else None


class Pipeline:
    def __init__(self, taxonomy: Taxonomy, ner: PresidioNer | None, normalize_asr: bool = True,
                 extra_ners: list | None = None, use_icd: bool = True):
        self.taxonomy = taxonomy
        self.ner = ner
        self.normalize_asr = normalize_asr
        # Bonus task "without ICD-10 codes at inference": the codes are then never read (the field is
        # still copied to the output); only the global clinical vocabulary protects clinical words.
        self.use_icd = use_icd
        self.extra_ners = extra_ners or []  # e.g. GlinerNer: same interface as PresidioNer

    @classmethod
    def from_config(cls, taxonomy_path: str | None = None, use_ner: bool = True,
                    use_truecasing: bool = True, normalize_asr: bool = True,
                    gliner_model: str | None = None, student_model: str | None = "auto",
                    use_icd: bool = True) -> Pipeline:
        """``student_model``: a model directory, "auto" (models/student if present) or None (off)."""
        taxonomy = Taxonomy.load(taxonomy_path or DEFAULT_TAXONOMY)
        ner = PresidioNer(use_truecasing=use_truecasing) if use_ner else None
        extra = []
        student = _load_student(student_model)
        if student:
            extra.append(student)
        if gliner_model:
            from clinanon.detectors.gliner_ner import GlinerNer  # optional dependency

            extra.append(GlinerNer(gliner_model, use_truecasing=use_truecasing))
        return cls(taxonomy, ner, normalize_asr=normalize_asr, extra_ners=extra, use_icd=use_icd)

    # ------------------------------------------------------------------------------------- detection

    def _candidates(self, texts: dict[str, str]) -> list[Span]:
        spans: list[Span] = []
        for field, text in texts.items():
            spans.extend(rules.detect(text, field))
            for ner in filter(None, [self.ner, *self.extra_ners]):
                spans.extend(ner.detect(text, field, reference=texts.get("note")))
        return [s for s in map(_first_line, spans) if s]

    def detect(self, record: dict) -> tuple[list[Span], list[tuple[Span, str]]]:
        """Final spans to act on, plus the vetoed candidates with their reasons."""
        texts = {f: record[f] for f in TEXT_FIELDS}
        protect = record_protect_terms(record) if self.use_icd else set()

        candidates, structural = _drop_structure(self._candidates(texts), texts)
        accepted, vetoed = veto.apply(candidates, texts, protect)
        names = propagation.collect_names(accepted, texts)
        places = propagation.collect_places(accepted, texts)
        mentions, structural_prop = _drop_structure(
            propagation.find_mentions(texts, names) + propagation.find_place_mentions(texts, places), texts
        )
        propagated, vetoed_prop = veto.apply(mentions, texts, protect)

        final = resolve_overlaps(accepted + propagated)
        # Spans found without a role (e.g. by NER) inherit the role of the same name found elsewhere.
        final = [
            dc_replace(s, role=names.get(s.text.lower())) if s.category == "NAME" and s.role is None else s
            for s in final
        ]
        return final, structural + vetoed + structural_prop + vetoed_prop

    # ----------------------------------------------------------------------------------- replacement

    def anonymize(self, record: dict) -> tuple[dict, list[Decision]]:
        """Anonymized copy of ``record`` and the decision log (offsets refer to the normalized text)."""
        if self.normalize_asr:
            record = {**record, **{f: normalize_asr_splits(record[f]) for f in TEXT_FIELDS}}
        spans, vetoed = self.detect(record)
        out = dict(record)  # encounter_id, dataset and icd10_codes pass through untouched
        decisions: list[Decision] = []
        eid = record.get("encounter_id", "")

        for field in TEXT_FIELDS:
            text = record[field]
            edits: list[tuple[int, int, str]] = []  # (start, end, replacement), ascending
            for span in (s for s in spans if s.field == field):
                action = self.taxonomy.action(span.category)
                replacement = self.taxonomy.replacement(span.category, span.role) if action == "tag" else None
                decisions.append(Decision(eid, field, span.start, span.end, span.text, span.category, action,
                                          replacement, span.source,
                                          f"{span.category} detected by {span.source}; policy: {action}",
                                          explain(span, action, replacement, self.taxonomy), _confidence(span)))
                if replacement is None:
                    continue
                # "[PATIENT] [PATIENT]" (first and last name found separately) -> one "[PATIENT]",
                # but never across a line break (that would merge two dialogue turns)
                gap = text[edits[-1][1] : span.start] if edits else ""
                if edits and edits[-1][2] == replacement and not gap.strip() and "\n" not in gap:
                    edits[-1] = (edits[-1][0], span.end, replacement)
                else:
                    edits.append((span.start, span.end, replacement))
            for start, end, replacement in reversed(edits):
                text = text[:start] + replacement + text[end:]
            out[field] = text

        for span, reason in vetoed:
            removed_by = next((evidence(s.source, s.category) for s in spans if s.overlaps(span)
                               and self.taxonomy.action(s.category) == "tag"), None)
            decisions.append(Decision(eid, span.field, span.start, span.end, span.text, span.category, "veto",
                                      None, span.source, reason, explain_veto(span, reason, removed_by),
                                      _confidence(span)))
        return out, decisions

    def anonymize_all(self, records: list[dict]) -> tuple[list[dict], list[Decision]]:
        outputs, decisions = [], []
        for record in records:
            out, dec = self.anonymize(record)
            outputs.append(out)
            decisions.extend(dec)
        return outputs, decisions
