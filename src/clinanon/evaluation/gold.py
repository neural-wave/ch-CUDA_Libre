"""Ground-truth identifiers from the ACI-Bench metadata (patient first/last name, age; doctor name for some).

The metadata covers all 207 encounters and is independent of our detectors, which makes it the most
trustworthy (if incomplete) privacy check we have: it says nothing about relatives, places or dates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_METADATA_DIR = Path(__file__).resolve().parents[3] / "data" / "external" / "aci_bench_metadata"


@dataclass(frozen=True)
class GoldIdentifiers:
    first_name: str | None
    last_name: str | None
    age: str | None
    doctor_name: str | None


def _normalize_age(raw: str) -> str | None:
    """The aci subset stores ages as floats ("58.0"); the text says "58-year-old"."""
    raw = raw.strip()
    try:
        return str(int(float(raw))) if raw else None
    except ValueError:
        return raw


def load_gold(metadata_dir: str | Path = DEFAULT_METADATA_DIR) -> dict[str, GoldIdentifiers]:
    gold: dict[str, GoldIdentifiers] = {}
    for path in sorted(Path(metadata_dir).glob("*.csv")):
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gold[row["encounter_id"]] = GoldIdentifiers(
                    first_name=row["patient_firstname"].strip() or None,
                    last_name=row["patient_familyname"].strip() or None,
                    age=_normalize_age(row["patient_age"]),
                    doctor_name=row["doctor_name"].strip() or None,
                )
    return gold
