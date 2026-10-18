"""Tests for the teacher's tag parsing and the ASR conversion of generated encounters (no model needed)."""

import importlib.util
from pathlib import Path

from clinanon.distill.teacher import parse_tagged

_spec = importlib.util.spec_from_file_location(
    "teacher_data", Path(__file__).resolve().parents[1] / "scripts" / "teacher_data.py")
teacher_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teacher_data)


def test_parse_tagged_gives_plain_text_and_exact_spans():
    tagged = "[doctor] hi <NAME>anna müller</NAME> , born <DATE>14 May 1958</DATE> . call <CONTACT>079 123 45 67</CONTACT>"
    plain, spans = parse_tagged(tagged)
    assert plain == "[doctor] hi anna müller , born 14 May 1958 . call 079 123 45 67"
    assert [(plain[s.start : s.end], s.category) for s in spans] == [
        ("anna müller", "NAME"), ("14 May 1958", "DATE"), ("079 123 45 67", "CONTACT")]


def test_parse_tagged_drops_broken_and_nested_tags():
    plain, spans = parse_tagged("seen at <ORGANIZATION>the <LOCATION>Lugano</LOCATION> clinic</ORGANIZATION> <NAME>Bob")
    assert plain == "seen at the Lugano clinic Bob"
    assert all(plain[s.start : s.end] == s.text for s in spans)


def test_policy_drops_relative_time_and_pronouns():
    assert not teacher_data.keep_span("DATE", "last week")
    assert not teacher_data.keep_span("NAME", "she")
    assert teacher_data.keep_span("DATE", "this summer")
    assert teacher_data.keep_span("TIME", "half past eight")


def test_asr_conversion_moves_spans():
    text = "[patient] I live in Zürich, at Bahnhofstrasse 12. Call me!"
    spans = [{"field": "dialogue", "start": 20, "end": 26, "text": "Zürich", "category": "LOCATION"},
             {"field": "dialogue", "start": 31, "end": 47, "text": "Bahnhofstrasse 12", "category": "LOCATION"}]
    spans[1]["end"] = spans[1]["start"] + len("Bahnhofstrasse 12")
    out, moved = teacher_data.to_asr(text, spans)
    assert out == "[patient] i live in zürich , at bahnhofstrasse 12 . call me !"
    assert [out[s["start"] : s["end"]] for s in moved] == ["zürich", "bahnhofstrasse 12"]


def test_student_spans_cover_whole_words_and_skip_format_categories():
    from clinanon.detectors.student import CONTEXT_CATEGORIES, StudentNer

    text = "SOCIAL HISTORY\n\nPatient reports he is a farmer. His ALT was 128."
    spans = []
    start = text.index("farm")
    StudentNer._close([start, start + 4, "PROFESSION", [0.9]], spans, text, "note")
    assert spans[0].text == "farmer"  # reaches the veto as a word (farmer's lung), not as "farm"

    student = object.__new__(StudentNer)
    student.categories, student.threshold = CONTEXT_CATEGORIES, 0.75
    age = text.index("128")
    StudentNer._close([age, age + 3, "AGE", [0.99]], spans, text, "note")
    assert not student._keep(spans[-1])  # lab values are not ages; formats are left to the rules
    StudentNer._close([0, 4, "NAME", [0.99]], spans, "damn it", "dialogue")
    assert not student._keep(spans[-1])
    StudentNer._close([0, 6, "LOCATION", [0.6]], spans, "carbo portion", "dialogue")
    assert not student._keep(spans[-1])


def test_missing_student_falls_back_with_a_warning(monkeypatch, tmp_path, caplog):
    from clinanon import pipeline
    from clinanon.detectors import student

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(student, "DEFAULT_MODEL", tmp_path / "no-model")
    monkeypatch.setattr(student, "HUB_MODEL", "clinanon/does-not-exist")
    assert pipeline._load_student("auto") is None
    assert "running rules + spaCy only" in caplog.text
