"""Lexicons shipped with the package (built by scripts/build_resources.py, loaded lazily and cached)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

_DIR = Path(__file__).resolve().parent


def _read_list(name: str) -> set[str]:
    lines = (_DIR / name).read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _read_tsv(name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in (_DIR / name).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("\t")
            out[key] = float(value)
    return out


@cache
def first_names() -> dict[str, float]:
    """First name -> max population share (0..1)."""
    return _read_tsv("first_names.tsv")


@cache
def surnames() -> dict[str, float]:
    """Surname -> number of bearers in the 2010 US census."""
    return _read_tsv("surnames.tsv")


@cache
def drug_tokens() -> set[str]:
    """FDA drug-name tokens, plus the parts of hyphenated ones ("toprol-xl" -> "toprol")."""
    tokens = _read_list("drug_tokens.txt")
    common = _read_list("common_words.txt") | set(STOP_WORDS)
    parts = {p for t in tokens if "-" in t for p in t.split("-") if len(p) >= 5 and p not in common}
    return tokens | parts


@cache
def icd_vocab() -> dict[str, float]:
    return _read_tsv("icd10cm_vocab.tsv")


@cache
def eponyms() -> set[str]:
    return set(_read_tsv("icd10cm_eponyms.tsv")) | curated_clinical_terms()


@cache
def curated_clinical_terms() -> set[str]:
    return _read_list("clinical_terms_curated.txt")


@cache
def occupations() -> set[str]:
    return _read_list("occupations.txt")


@cache
def places() -> frozenset[str]:
    """Lower-case place names (GeoNames cities >= 15,000 inhabitants, countries, regions; filtered)."""
    return frozenset(_read_list("places.txt"))


@cache
def common_words() -> set[str]:
    """Ordinary lower-case English words (as written in clinical notes) plus spaCy stop words."""
    return _read_list("common_words.txt") | set(STOP_WORDS)


@cache
def clinical_vocab() -> set[str]:
    """Words that carry clinical meaning: ICD-10-CM vocabulary + drugs + curated terms, minus stop words.

    Used by the clinical veto and by the utility proxy (clinical-token retention).
    """
    icd = {w for w, count in icd_vocab().items() if count >= 2}
    return (icd | drug_tokens() | curated_clinical_terms()) - set(STOP_WORDS)


def is_known_first_name(token: str, min_share: float = 0.0) -> bool:
    return first_names().get(token.lower(), -1.0) >= min_share


def is_known_surname(token: str, min_count: float = 100.0) -> bool:
    return surnames().get(token.lower(), -1.0) >= min_count
