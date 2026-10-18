"""Privacy and utility metrics comparing original and anonymized encounters.

Privacy
  * gold_privacy   - share of ground-truth identifier mentions (ACI metadata) removed.
  * judge_privacy  - proxy for the organizers' metric, under two readings of "proportion removed":
                     value survival (which values the judge detects in the original are still present in
                     the output) and detection count (1 - detections in output / detections in original).
Utility (cheap proxies; scripts/icd_check.py estimates ICD-coding F1 with an LLM coder)
  * icd_term_retention       - occurrences of the record's ICD-10 description words still present.
  * clinical_token_retention - occurrences of clinical-vocabulary tokens (ICD-10-CM, drugs) still present.
"""

from __future__ import annotations

import re
from collections import Counter

from clinanon import resources
from clinanon.evaluation.gold import GoldIdentifiers
from clinanon.evaluation.judges import Judge
from clinanon.types import TEXT_FIELDS

_TOKEN = re.compile(r"[a-z][a-z'-]*[a-z]")
# Words of ICD descriptions that say nothing about the diagnosis itself.
_ICD_GENERIC = {
    "unspecified", "without", "other", "specified", "initial", "encounter", "subsequent", "sequela",
    "elsewhere", "classified", "site", "part", "unknown", "type", "region", "complications", "sites",
}  # fmt: skip


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def count_mentions(text: str, value: str) -> int:
    """Case-insensitive occurrences of ``value`` not glued to other letters/digits."""
    pattern = rf"(?<![a-z0-9]){re.escape(_norm(value))}(?![a-z0-9])"
    return len(re.findall(pattern, _norm(text)))


# Detections no sensible scorer would count as private information: speaker tags ("[doctor]"), our own
# placeholders ("[PATIENT]") and pronouns or role words. Zero-shot judges flag these all the time.
_BRACKETED = re.compile(r"\[[^\]\n]*\]?|\]")
_PRONOUNS_AND_ROLES = {
    "i", "me", "you", "we", "he", "she", "they", "him", "her", "them", "patient", "the patient", "doctor",
    "the doctor", "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "nurse",
}  # fmt: skip


def is_trivial(text: str) -> bool:
    rest = _BRACKETED.sub(" ", text).strip(" \t\n.,;:-'\"").lower()
    return not re.search(r"[a-z0-9]", rest) or rest in _PRONOUNS_AND_ROLES


def _detect(judge: Judge, text: str) -> list:
    return [e for e in judge.detect(text) if not is_trivial(e.text)]


def _ratio(removed: float, total: float) -> float | None:
    return round(100.0 * removed / total, 1) if total else None


def gold_privacy(originals: list[dict], anonymized: list[dict], gold: dict[str, GoldIdentifiers]) -> dict:
    stats: Counter[str] = Counter()
    leaks: list[tuple[str, str, str]] = []
    for orig, anon in zip(originals, anonymized, strict=True):
        g = gold.get(orig["encounter_id"])
        if g is None:
            continue
        for field in TEXT_FIELDS:
            for kind, value in (("first_name", g.first_name), ("last_name", g.last_name)):
                if not value:
                    continue
                n_orig, n_left = count_mentions(orig[field], value), count_mentions(anon[field], value)
                stats[f"{kind}|{field}|orig"] += n_orig
                stats[f"{kind}|{field}|left"] += min(n_left, n_orig)
                if n_left:
                    leaks.append((orig["encounter_id"], field, value))
            if g.age:
                age_pattern = rf"\b{g.age}(?:[- ]?years?[- ]old|[- ]?y/?o\b| years? of age)"
                n_orig = len(re.findall(age_pattern, orig[field], flags=re.I))
                n_left = len(re.findall(age_pattern, anon[field], flags=re.I))
                stats[f"age|{field}|orig"] += n_orig
                stats[f"age|{field}|left"] += min(n_left, n_orig)
    table = {}
    for kind in ("first_name", "last_name", "age"):
        for field in TEXT_FIELDS:
            total, left = stats[f"{kind}|{field}|orig"], stats[f"{kind}|{field}|left"]
            table[f"{kind}/{field}"] = {"mentions": total, "removed_%": _ratio(total - left, total)}
    total = sum(v for k, v in stats.items() if k.endswith("|orig"))
    left = sum(v for k, v in stats.items() if k.endswith("|left"))
    return {"overall_removed_%": _ratio(total - left, total), "by_kind": table, "leaks": leaks}


def judge_privacy(originals: list[dict], anonymized: list[dict], judge: Judge) -> dict:
    by_label: Counter[str] = Counter()
    left_by_label: Counter[str] = Counter()
    residual: Counter[str] = Counter()
    survivors: Counter[str] = Counter()
    for orig, anon in zip(originals, anonymized, strict=True):
        for field in TEXT_FIELDS:
            values: Counter[tuple[str, str]] = Counter(
                (e.label, _norm(e.text)) for e in _detect(judge, orig[field])
            )
            for (label, value), n_orig in values.items():
                n_left = min(n_orig, count_mentions(anon[field], value))
                by_label[label] += n_orig
                left_by_label[label] += n_left
                if n_left:
                    survivors[f"{label}: {value}"] += n_left
            residual.update(e.label for e in _detect(judge, anon[field]))
    total, left = sum(by_label.values()), sum(left_by_label.values())
    return {
        "judge": judge.name,
        "removed_%": _ratio(total - left, total),
        "count_removed_%": _ratio(max(total - sum(residual.values()), 0), total),
        "detected_in_original": total,
        "residual_detections_in_output": sum(residual.values()),
        "residual_by_label": dict(residual.most_common()),
        "by_label": {
            label: {"detected": n, "removed_%": _ratio(n - left_by_label[label], n)}
            for label, n in by_label.most_common()
        },
        "top_survivors": survivors.most_common(25),
    }


def utility(originals: list[dict], anonymized: list[dict]) -> dict:
    clinical = resources.clinical_vocab()
    icd_orig = icd_left = clin_orig = clin_left = 0
    for orig, anon in zip(originals, anonymized, strict=True):
        icd_terms = {
            t
            for code in orig["icd10_codes"]
            for t in _TOKEN.findall(code["description"].lower())
            if len(t) >= 4 and t not in _ICD_GENERIC
        }
        for field in TEXT_FIELDS:
            o_tokens = Counter(_TOKEN.findall(orig[field].lower()))
            a_tokens = Counter(_TOKEN.findall(anon[field].lower()))
            for term in icd_terms:
                icd_orig += o_tokens[term]
                icd_left += min(o_tokens[term], a_tokens[term])
            for tok, n in o_tokens.items():
                if tok in clinical:
                    clin_orig += n
                    clin_left += min(n, a_tokens[tok])
    return {
        "icd_term_retention_%": _ratio(icd_left, icd_orig),
        "clinical_token_retention_%": _ratio(clin_left, clin_orig),
        "clinical_tokens_lost": clin_orig - clin_left,
    }
