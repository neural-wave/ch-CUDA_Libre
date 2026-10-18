"""Rule detectors: what must be found, and clinical look-alikes that must not be."""

import pytest

from clinanon.detectors import rules


def found(text: str, field: str = "dialogue") -> set[tuple[str, str]]:
    return {(s.category, s.text.lower()) for s in rules.detect(text, field)}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[doctor] hi mr . james , how are you ?", ("NAME", "james")),  # ASR splits "mr."
        ("[doctor] hi , i'm doctor isabella .", ("NAME", "isabella")),
        ("[doctor] next patient is randy gutierrez . date of birth , 8/10/2020 .", ("NAME", "randy gutierrez")),
        ("[doctor] eugene walker , n- date of birth 4/14/1960 .", ("NAME", "eugene walker")),  # surname = word
        ("[doctor] so sophia i see that you hurt your knee", ("NAME", "sophia")),  # "so" must not eat "so|phia"
        ("[doctor] mister thompson is a 67 -year-old male", ("NAME", "thompson")),  # name = clinical eponym
        ("[doctor] so hey there mark nice to see you again", ("NAME", "mark")),  # name = ordinary word
        ("[doctor] well good morning emily how are you", ("NAME", "emily")),
        ("[doctor] mrn 3-8-4-9-2-0 . she is here", ("IDENTIFIER", "3-8-4-9-2-0")),
        ("[patient] i'm 44 now and i smoke", ("AGE", "44")),
        ("[doctor] he died in his 40s .", ("AGE", "in his 40s")),
        ("[doctor] second week of june .", ("DATE", "june")),
    ],
)
def test_detects_identifiers_in_lowercase_dialogue(text, expected):
    assert expected in found(text)


def test_detects_identifiers_in_cased_note():
    note = ("HISTORY OF PRESENT ILLNESS\n\nSophia Brown is a 75 y.o. female. Seen by Dr. Reyes on 03/22/2021. "
            "Lives at Bahnhofstrasse 12. AHV 756.1234.5678.97.")
    hits = found(note, "note")
    assert ("NAME", "sophia brown") in hits
    assert ("NAME", "reyes") in hits
    assert ("DATE", "03/22/2021") in hits
    assert ("LOCATION", "bahnhofstrasse 12") in hits


def test_section_header_is_not_part_of_a_name():
    note = "ASSESSMENT AND PLAN\n\nMartha Collins is a 50-year-old female."
    names = {t for c, t in found(note, "note") if c == "NAME"}
    assert names == {"martha collins"}


@pytest.mark.parametrize(
    "text",
    [
        "[doctor] there is a slight 2/6 systolic ejection murmur",  # murmur grade
        "[patient] the pain is 7/10 today",  # pain score
        "[doctor] blood pressure 120/80",
        "[doctor] option 1 may be better",  # "may" is a verb
        "[doctor] keep sodium under <2000 mg per day",  # dose, not a year
        "[patient] i miss work sometimes",  # "miss" is a verb
        "[doctor] the doctor said it was fine",
        "[doctor] positive spurling's on the left",
    ],
)
def test_ignores_clinical_look_alikes(text):
    assert not {c for c, _ in found(text)} & {"NAME", "DATE", "IDENTIFIER"}
