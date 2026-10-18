"""End-to-end behaviour of the pipeline on small synthetic encounters."""

import pytest

from clinanon.pipeline import Pipeline
from clinanon.taxonomy import Taxonomy


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline.from_config(student_model=None)  # loads spaCy once for the module


def make_record(dialogue: str, note: str, codes=None) -> dict:
    return {
        "encounter_id": "T001",
        "dataset": "test",
        "dialogue": dialogue,
        "note": note,
        "icd10_codes": codes or [{"code": "G20", "description": "Parkinson's disease"}],
    }


def test_taxonomy_loads_and_tags_are_brackets():
    taxonomy = Taxonomy.load()
    for category in taxonomy.categories.values():
        assert category.tag.startswith("[") and category.tag.endswith("]")


def test_names_found_in_note_are_removed_from_dialogue(pipeline):
    record = make_record(
        dialogue="[doctor] hi , andrew . how are you ?\n[patient] good .\n[doctor] so , andrew , any tremor ?",
        note="HISTORY OF PRESENT ILLNESS\n\nAndrew Campbell is a 59-year-old male with Parkinson's disease.",
    )
    out, _ = pipeline.anonymize(record)
    assert "andrew" not in out["dialogue"].lower()
    assert "campbell" not in out["note"].lower()
    assert "59" not in out["note"]
    assert "Parkinson's disease" in out["note"]  # the diagnosis stays
    assert out["icd10_codes"] == record["icd10_codes"]  # codes are never touched


def test_structure_and_clinical_terms_are_preserved(pipeline):
    record = make_record(
        dialogue="[doctor] i'm gon na start mobic 15 milligrams .\n[patient] okay .\n"
                 "[doctor] you have a positive spurling's sign and a 2/6 murmur .",
        note="PLAN\n\nStart Mobic 15 mg daily.",
    )
    out, _ = pipeline.anonymize(record)
    assert out["dialogue"].startswith("[doctor]") and "\n[patient] okay ." in out["dialogue"]
    for term in ("mobic 15 milligrams", "spurling's", "2/6"):
        assert term in out["dialogue"]
    assert out["note"] == record["note"]


def test_patient_named_like_a_disease(pipeline):
    record = make_record(
        dialogue="[doctor] good morning , mr. parkinson .\n[doctor] parkinson's disease runs in your family ?",
        note="HISTORY OF PRESENT ILLNESS\n\nMr. Parkinson is a 70-year-old male.",
    )
    out, _ = pipeline.anonymize(record)
    assert "mr. parkinson" not in out["dialogue"].lower()
    assert "parkinson's disease" in out["dialogue"]


def test_every_decision_has_a_reason(pipeline):
    record = make_record(dialogue="[doctor] hi , andrew .", note="Andrew Campbell is a 59-year-old male.")
    _, decisions = pipeline.anonymize(record)
    assert decisions and all(d.reason for d in decisions)
