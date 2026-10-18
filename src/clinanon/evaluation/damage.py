"""Damage audit: list every edit the anonymizer made, flagging the ones that look clinical.

This is the most actionable debugging view we have: proxies say *that* utility dropped, the audit says
*which words* were removed ("Mobic -> [PERSON]"). Read the flagged edits after every pipeline change.
"""

from __future__ import annotations

import csv
import difflib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from clinanon import resources
from clinanon.evaluation.gold import GoldIdentifiers
from clinanon.types import TEXT_FIELDS

_WORD = re.compile(r"[a-z][a-z'-]*[a-z]")


@dataclass
class Edit:
    encounter_id: str
    field: str
    original: str
    replacement: str
    context: str
    suspected_clinical: bool


def _looks_clinical(span: str, gold_names: set[str]) -> bool:
    """A removed span is suspicious if it contains a clinical word that is not also a person's name."""
    clinical = resources.clinical_vocab()
    for tok in _WORD.findall(span.lower()):
        tok = tok.removesuffix("'s")
        if tok in gold_names:
            return False
        is_name = resources.is_known_first_name(tok, 1e-4) or resources.is_known_surname(tok, 1000)
        if tok in clinical and (not is_name or tok in resources.drug_tokens()):
            return True
    return False


def audit(originals: list[dict], anonymized: list[dict], gold: dict[str, GoldIdentifiers]) -> list[Edit]:
    edits: list[Edit] = []
    for orig, anon in zip(originals, anonymized, strict=True):
        g = gold.get(orig["encounter_id"])
        gold_names = {n.lower() for n in (g.first_name, g.last_name) if n} if g else set()
        for field in TEXT_FIELDS:
            a, b = orig[field].split(), anon[field].split()
            matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == "equal" or i1 == i2:
                    continue
                span = " ".join(a[i1:i2])
                edits.append(
                    Edit(
                        encounter_id=orig["encounter_id"],
                        field=field,
                        original=span,
                        replacement=" ".join(b[j1:j2]),
                        context=" ".join(a[max(0, i1 - 6) : i2 + 6]),
                        suspected_clinical=_looks_clinical(span, gold_names),
                    )
                )
    return edits


def summarize(edits: list[Edit], top: int = 30) -> dict:
    flagged = [e for e in edits if e.suspected_clinical]
    return {
        "edits": len(edits),
        "suspected_clinical": len(flagged),
        "top_suspected": Counter(f"{e.original} -> {e.replacement}" for e in flagged).most_common(top),
    }


def write_csv(edits: list[Edit], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Edit.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(e) for e in edits)
