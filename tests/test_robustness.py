"""Formats that ACI-Bench barely contains but other data may: European phone numbers, IDs,
addresses and hospitals, spoken contact data, employers, and named relatives.

The positive cases are written in the style of the synthetic dev split (scripts/make_synthetic.py); the
negative cases are the false positives that these rules first produced on the real ACI encounters.
"""

import pytest

from clinanon.detectors.rules import detect
from clinanon.pipeline import Pipeline


def found(text: str, field: str = "dialogue") -> set[tuple[str, str]]:
    return {(s.category, s.text.lower()) for s in detect(text, field)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sure, here's my phone number 081 824 08 12.", ("CONTACT", "081 824 08 12")),
        ("I see you have +41 58 879 29 45 on file", ("CONTACT", "+41 58 879 29 45")),
        ("Phone: +44(0)117 496 0081", ("CONTACT", "+44(0)117 496 0081")),
        ("Your primary phone is (07243) 808264 and", ("CONTACT", "(07243) 808264")),
        ("call 001-534-913-4990x2284 tomorrow", ("CONTACT", "001-534-913-4990x2284")),
        ("we have your zero six zero eight three zero one six two one on file", ("CONTACT", "zero six zero eight three zero one six two one")),
        ("backup email is branka dot braun at hotmail dot com .", ("CONTACT", "branka dot braun at hotmail dot com")),
        ("Insurance: CSS, policy no. 6918 8981 8295", ("IDENTIFIER", "6918 8981 8295")),
        ("Here it is: 8075679768636.", ("IDENTIFIER", "8075679768636")),
        ("look up my sympany and FW6410349 in the system", ("IDENTIFIER", "fw6410349")),
        ("Patient ID: MRN-065860", ("IDENTIFIER", "065860")),
        ("Patient ID: PID 27907735", ("IDENTIFIER", "27907735")),
        ("AHV: 756.1234.5678.97", ("IDENTIFIER", "756.1234.5678.97")),
        ("it's rue patrick saudan 96 in 9570 st. gallen , correct ?", ("LOCATION", "rue patrick saudan 96")),
        ("it's rue patrick saudan 96 in 9570 st. gallen , correct ?", ("LOCATION", "9570 st. gallen")),
        ("where i live , corinne-schwital-straße 4 , to arrange", ("LOCATION", "corinne-schwital-straße 4")),
        ("Here it is: chemin Joye 90, 2054 Bern.", ("LOCATION", "chemin joye 90")),
        ("Can you confirm the 7886 Sion as well?", ("LOCATION", "7886 sion")),
        ("the postcode LS7 6FH Brighton.", ("LOCATION", "ls7 6fh brighton")),
        ("i just relocated to thun from the united states", ("LOCATION", "united states")),
        ("i saw dr . gadda at kantonsspital st. gallen from may", ("ORGANIZATION", "kantonsspital st. gallen")),
        ("referred to Dr. Montesano at Inselspital Bern for testing.", ("ORGANIZATION", "inselspital bern")),
        ("consult Dr. Acerbi at Klinik Hirslanden if symptoms worsen", ("ORGANIZATION", "klinik hirslanden")),
        ("my consultation at ospedale regionale di lugano was", ("ORGANIZATION", "ospedale regionale di lugano")),
        ("a referral to Dr. Daly at Leeds General Infirmary.", ("ORGANIZATION", "leeds general infirmary")),
        ("She works part time as a tram driver at Blum & Cie.", ("ORGANIZATION", "blum & cie")),
        ("Profession is architect, employer is Brun Hug GmbH.", ("ORGANIZATION", "brun hug gmbh")),
        ("he works as a flight attendant for White LLC.", ("ORGANIZATION", "white llc")),
        ("uh , you worked with enterprise rent-a-car and one of", ("ORGANIZATION", "enterprise rent-a-car")),
        ("She works part time as a tram driver at Blum & Cie.", ("PROFESSION", "tram driver")),
        ("i work as a computer programmer and i've been", ("PROFESSION", "computer programmer")),
        ("You mentioned she is a primary school teacher at the", ("PROFESSION", "primary school teacher")),
        ("let's lock in the 20th of July at 15.15.", ("TIME", "15.15")),
        ("please book evéquoz for december 18 at half past eight .", ("TIME", "half past eight")),
        ("seen on the twenty-ninth of march .", ("DATE", "the twenty-ninth of march")),
        ("born may eleventh nineteen sixty , right ?", ("DATE", "may eleventh")),
        ("actually , elizabeth , who is 77 , is going to stay", ("AGE", "77")),
        ("calogero , 35 , was the one who found out", ("AGE", "35")),
        ("My aunt, Rembrandt, who is exactly 22 years old", ("NAME", "rembrandt")),
        ("my mom , aurélie , who is 61 , comes over", ("NAME", "aurélie")),
        ("Actually, Beppe, who is 54, is going to stay with me", ("NAME", "beppe")),
    ],
)
def test_detects_formats_outside_aci(text, expected):
    assert expected in found(text, "note" if text[0].isupper() else "dialogue")


@pytest.mark.parametrize(
    "text",
    [
        "um , my cousin said , thankfully , my children are fine . well , actually , my husband",
        "she recently learned , her partner of ten years",
        "he had positive lab work for hep c so how're you",
        "it only works for the barky cough",
        "the medication is working for me",
        "why did you check these levels in the first place",
        "we can consider hand clinic referral for injections",
        "i'm on a bunch of meds . soak your feet in epsom salts",
        "blood sugars were 140 150 160 , bp 120/80 , 1000 calories",
        "a 2/6 murmur , pain 7/10 , platelets 250,000 , weight 70.5 kg at 1.25 mg",
        "take 1.25 mg at 8 , via email , via iv , route of administration",
        "the plan is in 2 weeks , you may first try ice",
    ],
)
def test_ignores_look_alikes_from_real_encounters(text):
    categories = {"NAME", "ORGANIZATION", "LOCATION", "CONTACT", "IDENTIFIER", "TIME", "PROFESSION", "AGE"}
    assert not {f for f in found(text) if f[0] in categories}


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline.from_config(student_model=None)


def test_longer_organization_beats_gazetteer_place(pipeline):
    record = {"encounter_id": "T1", "dataset": "t", "dialogue": "[doctor] hi .",
              "note": "SOCIAL HISTORY\n\nPatient is a fan of the Denver Nuggets basketball team.",
              "icd10_codes": []}  # fmt: skip
    out, _ = pipeline.anonymize(record)
    assert "Nuggets" not in out["note"] and "basketball team" in out["note"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PATIENT: SILVA, MARIA\nMRN: CH-78421-99", ("NAME", "silva, maria")),
        ("PATIENT: SILVA, MARIA\nMRN: CH-78421-99", ("IDENTIFIER", "ch-78421-99")),
        ("DATE OF BIRTH: 14-MAY-1960 (Age: 75)", ("DATE", "14-may-1960")),
        ("DATE OF BIRTH: 14-MAY-1960 (Age: 75)", ("AGE", "75")),
        ("INS NO: 74-992-104\nINSURER: HELSANA", ("ORGANIZATION", "helsana")),
        ("ASSISTANTS: DR. M. ROSTI, N. BERG\nNURSE: J. VOGEL, RN", ("NAME", "j. vogel")),
        ("His brother Ravi (30) has diabetes.", ("AGE", "30")),
    ],
)
def test_detects_hospital_form_fields(text, expected):
    assert expected in found(text, "note")


def test_form_fields_ignore_clinical_values():
    assert not found("Service: Internal Medicine\nPatient: alert and oriented, in no distress.", "note")


def test_drugs_food_and_clinical_terms_used_as_names_are_kept(pipeline):
    record = {"encounter_id": "T1", "dataset": "t", "icd10_codes": [],
              "note": "Continue Toprol 50 mg daily. He likes French fries.",
              "dialogue": "[patient] my neighbor is mrs . thrombosis .\n"
                          "[doctor] we need to rule out a deep vein thrombosis ."}  # fmt: skip
    out, decisions = pipeline.anonymize(record)
    assert "Toprol 50 mg" in out["note"] and "French fries" in out["note"]
    assert "mrs . [PERSON]" in out["dialogue"] and "deep vein thrombosis" in out["dialogue"]
    assert all(d.explanation for d in decisions)


def test_rejected_candidate_explains_when_span_is_removed_anyway(pipeline):
    record = {"encounter_id": "T1", "dataset": "t", "icd10_codes": [], "dialogue": "[doctor] hi .",
              "note": "Mr. Allen reports abdominal pain. Mr. Allen denies fever."}
    _, decisions = pipeline.anonymize(record)
    vetoes = [d for d in decisions if d.action == "veto" and d.text == "Allen"]
    assert all("removed anyway" in d.explanation for d in vetoes)


def test_form_labels_are_not_names_and_punctuation_ends_an_employer(pipeline):
    record = {"encounter_id": "T1", "dataset": "t", "icd10_codes": [],
              "note": "Patient: Keller, Anna\nDOB: 03-MAR-1961\nMRN: CH-48213-07",
              "dialogue": "[patient] i work as a tram driver for the sbb , my daughter drives me ."}
    out, _ = pipeline.anonymize(record)
    assert "DOB: [DATE]" in out["note"] and "MRN: [ID]" in out["note"]
    assert "the [ORGANIZATION] , my daughter" in out["dialogue"]
