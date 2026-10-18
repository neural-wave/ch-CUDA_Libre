"""Clinical veto: decide, per candidate span, whether it is clinical information rather than PII.

Only *weak* candidates can be vetoed. A candidate is weak if it comes from statistical NER or from an
ambiguous pattern (phone/URL). Strong spans (title + name, dictation header, dates, ages, IDs) are
trusted.

Two kinds of evidence, applied in this order:
1. Context (applies to every weak span, including names propagated across the record):
     "Parkinson's disease", "positive Spurling's", "Mobic 15 mg"         -> clinical
2. Vocabulary (applies only to NER-only spans, not to names already established in the record):
     every token is clinical (ICD-10-CM, FDA drugs, curated) or an ordinary word, and none is a
     common person name                                               -> clinical
   plus small rule sets for ASR fragments ("wan na"), relationship words ("dad") and common nouns
   introduced by a determiner ("a slight nick").

Every veto returns a human-readable reason, logged in decisions.jsonl.
"""

from __future__ import annotations

import re

from clinanon import resources
from clinanon.detectors.rules import GENERIC_FACILITY_WORDS, is_cased
from clinanon.types import Span

# Head nouns that turn a preceding eponym into a clinical concept.
_CLINICAL_HEAD_AFTER = re.compile(
    r"^(?:'s|’s|s'|s)?\s*-?\s*(?:disease|syndrome|sign|signs|test|tests|maneuver|manoeuvre|reflex|palsy|"
    r"tremor|lymphoma|sarcoma|thyroiditis|esophagus|contracture|cyst|ulcer|fracture|phenomenon|neuroma|"
    r"node|nodes|criteria|score|scale|classification|procedure|repair|deformity|disorder|dementia|"
    r"arthritis|titer|titers|reaction|tear|injury|position|tendon|tendinitis|tendinopathy|bursitis|"
    r"ataxia|chorea|anomaly|malformation|neuralgia|depth|thickness|level|stage|grade|index|method|"
    r"technique|incision|view|line|ligament|joint|complex|amputation|dislocation|cell|cells|coma)\b",
    re.I,
)
_CLINICAL_CONTEXT_BEFORE = re.compile(r"\b(?:positive|negative|equivocal|\+|-)\s*$", re.I)
# Dosing/formulation context around a drug name: "mobic 15 milligrams", "camila birth control".
_DRUG_CONTEXT_AFTER = re.compile(
    r"^\s*(?:\d|xr\b|er\b|sr\b|xl\b|hct\b|birth control|tablets?\b|pills?\b|cream|ointment|injections?|"
    r"inhaler|patch|drops|gel|daily|once|twice|milligrams?|mg\b)", re.I)
_DRUG_CONTEXT_BEFORE = re.compile(
    r"\b(?:taking|take|takes|took|on|started|start|prescribed|prescribe|stopped|stop|discontinued|"
    r"continue|refill|refilled|increase|decrease|using|use|recommend|recommended|try)\s+(?:the\s+|some\s+)?$",
    re.I)
# A dose right after a name: "Toprol 50 mg daily", "toprol fifty milligrams" (drugs missing from the FDA list).
_DOSE_AFTER = re.compile(
    r"^\s*(?:-?\s*xl\s*|-?\s*er\s*)?(?:\d+(?:\.\d+)?|(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|"
    r"twenty-five|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|one hundred)(?:[- ]\w+)?)\s*"
    r"(?:mg|mcg|milligrams?|micrograms?|units?|ml|milliliters?|grams?|g)\b", re.I)
_NOT_A_PERSON_AFTER = re.compile(r"^(?:['’]s)?\s*(?:fries|food|foods|cuisine|restaurant|toast|dressing|bread|cheese|"
                                 r"press|kiss|horn|door|style|dish|dishes|salad|rice|sauce)\b", re.I)
_TITLE_BEFORE = re.compile(r"\b(?:mr|mrs|ms|miss|mister|missus|dr|doctor|prof)\s*\.?\s*$", re.I)
_BLOOD_PRESSURE_LIKE = re.compile(r"^\s*\d{2,3}\s*/\s*\d{2,3}\s*$")
# ASR splits contractions into two tokens: "wan na", "gon na", "lem me". NER tags them as persons.
_ASR_SPLIT = re.compile(r"^(?:wan|gon|got|lem|kin|sort|out)(?:\s+(?:na|ta|me|da|a))?$", re.I)
_ASR_SPLIT_NEXT = re.compile(r"^\s*(?:na|ta|me|da|a)\b", re.I)
_DETERMINER_BEFORE = re.compile(r"\b(?:a|an|the|some|slight|small|little|any|no|this|that|your|his|her)\s+$",
                                re.I)
# Propagated names: only indefinite/quantifying determiners mark a common noun ("a slight nick");
# "the karen" occurs in garbled ASR and is still the patient.
_INDEFINITE_BEFORE = re.compile(r"\b(?:a|an|some|slight|small|little|any)\s+$", re.I)
# Propagated names that are also words ("mark", "brown", "walker"): any determiner, possessive or
# descriptive adjective before them means the word is used as a word.
_WORD_MODIFIER_BEFORE = re.compile(
    r"\b(?:a|an|the|my|your|his|her|their|our|its|this|that|these|those|some|any|no|own|dark|light|"
    r"bright|pale|red|small|large|little|big|new|old|birth|skin)\s+$", re.I)
# Facility names are identifying even though they contain ordinary words ("East Metropolitan Hospital").
_FACILITY = re.compile(r"\b(?:hospital|clinic|medical center|health center|healthcare|infirmary|hospice|"
                       r"regional|university|college|school|center|centre|practice)\b", re.I)
_MEDICAL_SUFFIX = re.compile(
    r"(?:itis|osis|emia|algia|ectomy|otomy|ostomy|plasty|scopy|graphy|gram|megaly|pathy|plasia|trophy|"
    r"cytic|otic|ary|eal|ular|ic)$")
RELATIONSHIP_WORDS = {
    "dad", "daddy", "mom", "mommy", "mother", "father", "wife", "husband", "son", "daughter", "brother",
    "sister", "grandma", "grandpa", "grandmother", "grandfather", "aunt", "uncle", "cousin", "baby",
    "honey", "sweetie", "buddy", "sir", "ma'am", "doc", "nurse", "doctor", "partner", "boyfriend",
    "girlfriend", "friend",
}  # fmt: skip
_VETOABLE = {"NAME", "LOCATION", "ORGANIZATION", "DEMOGRAPHIC", "CONTACT", "PROFESSION"}
# Tokens for vocabulary look-ups: hyphens and slashes split ("Toprol-XL", "Hematology/Lymphatic").
_TOKEN = re.compile(r"[^\W\d_][^\W\d_'’]*")


def _tokens(text: str) -> list[str]:
    return [t.lower().removesuffix("'s").removesuffix("’s") for t in _TOKEN.findall(text)]


def _is_common_person_name(token: str) -> bool:
    return resources.is_known_first_name(token, 1e-4) or resources.is_known_surname(token, 5000)


def is_known_name(token: str) -> bool:
    return resources.is_known_first_name(token, 1e-5) or resources.is_known_surname(token, 100)


def veto_reason(span: Span, text: str, record_terms: set[str]) -> str | None:
    """Reason why ``span`` is clinical/non-PII (so it must be kept), or None if it should be anonymized."""
    if span.strong or span.category not in _VETOABLE:
        return None
    before, after = text[max(0, span.start - 40) : span.start], text[span.end : span.end + 40]
    tokens = _tokens(span.text)

    if span.category == "PROFESSION":
        # Occupational exposure can be the diagnosis itself: "farmer" -> farmer's lung (J67.0).
        if any(t in record_terms or t in resources.eponyms() for t in tokens):
            return "occupation that is part of a diagnosis (ICD-10-CM eponym or the record's ICD terms)"
        return None

    # 1. Context -------------------------------------------------------------------------------------
    if span.category in {"NAME", "LOCATION", "ORGANIZATION"}:
        if _CLINICAL_HEAD_AFTER.match(after):
            return "it is followed by a clinical head noun (disease, sign, test, joint ...), so it names a clinical concept"
        if _CLINICAL_CONTEXT_BEFORE.search(before):
            return "clinical sign context (positive/negative ...)"
    if span.category in {"NAME", "ORGANIZATION", "LOCATION"} and _DOSE_AFTER.match(after):
        return "it is followed by a dose, so it is a drug"
    if span.category == "DEMOGRAPHIC" and _NOT_A_PERSON_AFTER.match(after):
        return "it describes food or a thing, not a person"
    if span.category == "CONTACT" and _BLOOD_PRESSURE_LIKE.match(span.text):
        return "measurement (looks like a blood pressure), not a phone number"
    if any(t in resources.drug_tokens() for t in tokens) and (
        _DRUG_CONTEXT_AFTER.match(after) or _DRUG_CONTEXT_BEFORE.search(before)
    ):
        return "drug name in a dosing/medication context"
    if span.source.startswith("gazetteer"):
        return None  # ordinary-word place names were filtered out when the list was built
    if span.source == "propagation":
        # The name is established elsewhere in the record; only context can overrule it. A name that is
        # also a clinical term ("my neighbour mrs. thrombosis") only propagates where a title marks it:
        # "deep vein thrombosis" stays clinical.
        if (span.category == "NAME" and tokens and all(t in resources.clinical_vocab() for t in tokens)
                and not any(_is_common_person_name(t) for t in tokens) and not _TITLE_BEFORE.search(before)):
            return "clinical term that is also a name elsewhere in the encounter; no title here, so clinical"
        if _INDEFINITE_BEFORE.search(before):
            return "common noun introduced by an indefinite determiner"
        if tokens and tokens[0] in resources.common_words() and _WORD_MODIFIER_BEFORE.search(before):
            return "name that is also a word, used as a word here (determiner/possessive/adjective before)"
        return None

    # 2. Vocabulary (NER-only spans) -------------------------------------------------------------------
    if not tokens:
        return "no alphabetic content"
    if all(t in RELATIONSHIP_WORDS for t in tokens):
        return "relationship word, not a name"
    if _ASR_SPLIT.match(span.text.strip()) and (" " in span.text.strip() or _ASR_SPLIT_NEXT.match(after)):
        return "ASR fragment (wan na / gon na / lem me)"
    if span.category == "NAME" and _DETERMINER_BEFORE.search(before):
        return "common noun introduced by a determiner"
    if span.text.isupper() and all(len(t) <= 5 for t in tokens):
        return "it is a clinical abbreviation or acronym"  # "IV", "PET CT"
    if span.category == "ORGANIZATION" and (
        re.search(r"[\d/]", span.text) or (len(tokens) == 1 and len(tokens[0]) <= 4)
    ):
        return "abbreviation or unit, not an organization"
    # The student was trained on European names missing from our (US) name lists, so it is exempt.
    if (span.category == "NAME" and not any(is_known_name(t) for t in tokens) and not span.source.startswith("ner:student")
            and (not is_cased(text) or len(tokens) == 1)):
        return "not a known person name (NER guess on a single/lower-cased token)"
    clinical = resources.clinical_vocab() | record_terms
    protected = resources.drug_tokens() | resources.curated_clinical_terms() | record_terms

    def clinical_or_common(t: str) -> bool:
        return t in clinical or t in resources.common_words() or bool(_MEDICAL_SUFFIX.search(t))

    has_person_name = any(_is_common_person_name(t) and t not in protected for t in tokens)
    facility = _FACILITY.search(span.text)
    # A clinical word is no evidence of a name here, even if it is also a surname ("Hand Clinic").
    named = any(_is_common_person_name(t) and t not in clinical and t not in GENERIC_FACILITY_WORDS for t in tokens)
    if span.category in {"ORGANIZATION", "LOCATION"} and facility and not named and all(
        t in clinical or t in GENERIC_FACILITY_WORDS or _FACILITY.fullmatch(t) for t in tokens
    ):
        return "it is a type of facility or department, not a named one"
    if all(clinical_or_common(t) for t in tokens) and not has_person_name:
        return "all its words are clinical (ICD-10-CM, FDA drug list) or ordinary vocabulary"
    if span.category in {"ORGANIZATION", "LOCATION"} and not facility:
        # Organizations/places named after products are drugs/devices ("Proair HFA"), and multi-word
        # "organizations" made mostly of clinical words are findings ("Target BP", "AC joints").
        if any(t in resources.drug_tokens() for t in tokens):
            return "drug/product name, not an organization"
        if sum(clinical_or_common(t) for t in tokens) * 2 >= len(tokens) and not has_person_name:
            return "mostly clinical or ordinary vocabulary"
    return None


def apply(spans: list[Span], texts: dict[str, str], record_terms: set[str]) -> tuple[list[Span], list[tuple[Span, str]]]:
    """Split candidates into (accepted, [(vetoed span, reason), ...])."""
    accepted, vetoed = [], []
    for span in spans:
        reason = veto_reason(span, texts[span.field], record_terms)
        if reason:
            vetoed.append((span, reason))
        else:
            accepted.append(span)
    return accepted, vetoed
