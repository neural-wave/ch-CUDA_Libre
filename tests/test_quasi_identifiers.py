"""Quasi-identifiers: calendar elements, clock times, places and facilities, occupations, vendors, ages.

The positive cases are misses found on the eval set (wording taken from the encounters); the negative
cases are the clinical look-alikes that must survive.
"""

import pytest

from clinanon.detectors.rules import detect
from clinanon.pipeline import Pipeline


def found(text: str) -> set[tuple[str, str]]:
    return {(s.category, s.text.lower()) for s in detect(text, "dialogue") if s.category != "NAME"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("post bilateral reduction mammoplasty on 10-10 20-20 .", ("DATE", "10-10 20-20")),
        ("who biopsied the 0.7 mm lesion on 04/10 and diagnosed the melanoma", ("DATE", "04/10")),
        ("i see you're scheduled on the 24th for surgery .", ("DATE", "24th")),
        ("it was brutal in november and december .", ("DATE", "december")),
        ("it's been hurting since about march .", ("DATE", "march")),
        ("okay , so this was march of this year , so maybe", ("DATE", "march of this year")),
        ("the same screening that you had for august first so", ("DATE", "august first")),
        ("you had a heart attack in twenty eighteen we put a stent", ("DATE", "twenty eighteen")),
        ("uh , saturday night it actually kept me up", ("DATE", "saturday")),
        ("next monday is actually my birthday", ("DATE", "monday")),
        ("having her right knee replaced this summer at east metro", ("DATE", "summer")),
        ("seven o'clock last night , so a little more than that", ("TIME", "seven o'clock")),
        ("i woke up about two o'clock in the morning", ("TIME", "two o'clock")),
        ("to be at work by 7:00 , lecture", ("TIME", "7:00")),
        ("he woke around 2:00 am with intense pain", ("TIME", "2:00 am")),
        ("a new restaurant down on fifth street", ("LOCATION", "fifth street")),
        ("the one down on main street", ("LOCATION", "main street")),
        ("like around the mount baker area", ("LOCATION", "mount baker")),
        ("some new trails here behind the rex center", ("ORGANIZATION", "rex center")),
        ("the other option is valley medical , and they do it", ("ORGANIZATION", "valley medical")),
        ("dr. martin who is at county surgical services down here", ("ORGANIZATION", "county surgical services")),
        ("Ms. Gutierrez is employed as an x-ray technician.", ("PROFESSION", "x-ray technician")),
        ("i own an auto repair shop .", ("PROFESSION", "auto repair shop")),
        ("don't they own that that bakery down on the corner", ("PROFESSION", "bakery")),
        ("i was in mount saint helen and washington", ("LOCATION", "mount saint helen")),
        ("it's the spring of twenty twenty", ("DATE", "twenty twenty")),
        ("hey dragon , ms. hernandez is a 48-year-old female", ("VENDOR", "dragon")),
        ("they reconstructed the whole thing i was fourteen i was a gymnast", ("AGE", "fourteen")),
        ("the injury that you had when you were fourteen i'm gonna", ("AGE", "fourteen")),
        ("i'm almost 40 and my mom was almost 45 when she was diagnosed", ("AGE", "40")),
        ("i'm almost 40 and my mom was almost 45 when she was diagnosed", ("AGE", "45")),
        ("i'd like to tell dax a little bit about you", ("VENDOR", "dax")),
        ("enter your findings at epic mychart and i'll get a reading", ("VENDOR", "epic mychart")),
        ("Send MyChart message in 1 month", ("VENDOR", "mychart")),
    ],
)
def test_detects_quasi_identifiers(text, expected):
    assert expected in found(text)


@pytest.mark.parametrize(
    "text",
    [
        "pain is 4/10 , was 7/10 , 2/6 systolic murmur , bp 120/80",
        "it went from 4/10 to 2/10 with ibuprofen",
        "swelling of the 5th digit , better after the 2nd time",
        "a mass at the two o'clock position , 11 centimeters from the nipple",
        "2:1 av block on the ekg",
        "which one am i supposed to take",
        "previous medical history , the infusion center , a cancer center",
        "the main road , 30 miles down the road , a 15 minute drive",
        "the vaccine can mount a response",
        "the fall injured her hip",
        "i'm doctor smith , this is a doctor's office",
        "twenty twenty vision , ice it for twenty thirty minutes at a time",
        "we went to the aquatic center and the wreck center",
        "chief complaint\n\nhospital follow-up",
        "dragon fruit",
        "i see on my chart here that you had a bunch of lower respiratory infections",
        "i'm about 200 pounds , he was 120 over 80 , i was one of the first",
        "i was five foot two , i'm 6 feet , the sugar was 145 when i checked",
        "She is about 1.5 kg down in weight. she's about 3 , 3 and a half pounds down",
        "because i'm eight years out from surgery , and",
    ],
)
def test_ignores_clinical_look_alikes(text):
    categories = {"DATE", "TIME", "LOCATION", "ORGANIZATION", "PROFESSION", "VENDOR", "AGE"}
    assert not {f for f in found(text) if f[0] in categories}


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return Pipeline.from_config(student_model=None)


def make_record(dialogue: str, note: str) -> dict:
    return {"encounter_id": "T1", "dataset": "t", "dialogue": dialogue, "note": note,
            "icd10_codes": [{"code": "J67.9", "description": "Hypersensitivity pneumonitis"}]}  # fmt: skip


def test_facility_named_in_note_is_removed_from_dialogue(pipeline):
    record = make_record(
        dialogue="[doctor] her knee replaced this summer at east metro .\n"
        "[doctor] he has been seen at point may orthopedics .\n[doctor] at that point may i ask ?",
        note="She wants the knee replaced at East Metropolitan Hospital. Seen at Point May Orthopedics.",
    )
    out, _ = pipeline.anonymize(record)
    assert "east metro" not in out["dialogue"] and "point may orthopedics" not in out["dialogue"]
    assert "at that point may i ask" in out["dialogue"]


def test_occupation_that_is_part_of_the_diagnosis_is_kept(pipeline):
    record = make_record(
        dialogue="[doctor] you mentioned that you are a farmer .\n[patient] my wife is a teacher .",
        note="He is a farmer and has been moving hay. His wife is employed as an x-ray technician.",
    )
    out, _ = pipeline.anonymize(record)
    assert "farmer" in out["dialogue"] and "farmer" in out["note"]  # farmer's lung (J67.0)
    assert "teacher" not in out["dialogue"] and "x-ray technician" not in out["note"]


def test_asr_contractions_are_rejoined(pipeline):
    out, _ = pipeline.anonymize(make_record("[doctor] i'm gon na check , lem me see .", "Exam."))
    assert "gonna" in out["dialogue"] and "lemme" in out["dialogue"]


def test_clinical_abbreviations_and_departments_are_not_organizations(pipeline):
    record = make_record(  # wording of D2N122 / D2N179, where NER tags both as organizations
        dialogue="[doctor] i'm going to schedule a pet ct and a referral to our hand clinic .",
        note="PET CT will also be scheduled to determine if the nodule is cancerous. If still symptomatic, "
        "we can consider a referral to our Hand Clinic possibly for injections or carpal tunnel release.",
    )
    out, decisions = pipeline.anonymize(record)
    assert "PET CT" in out["note"] and "Hand Clinic" in out["note"] and "hand clinic" in out["dialogue"]
    assert any(d.text == "Hand Clinic" and d.action == "veto" and "type of facility" in d.reason for d in decisions)
