"""Tests for the evaluation helpers and the teacher's date filter (no models are loaded)."""

from clinanon.evaluation.judges import chunks
from clinanon.evaluation.metrics import is_trivial


def test_chunks_keep_offsets_and_size():
    text = "[doctor] hi martha . how are you ?\n[patient] fine . " * 40 + "\nlast line without break"
    pieces = list(chunks(text, max_words=50))
    assert all(text[o : o + len(c)] == c for o, c in pieces)
    assert all(len(c.split()) <= 50 for _, c in pieces)
    assert pieces[-1][1].endswith("last line without break")


def test_chunks_never_split_inside_a_sentence():
    text = "Martha Collins is a 50-year-old female. She presents today. " * 30
    for _, chunk in chunks(text, max_words=20):
        assert chunk.rstrip().endswith(".")


def test_trivial_detections_are_ignored():
    for text in ["[doctor]", "[patient", "I", "you", "[PATIENT]", "the patient", "Dr."]:
        assert is_trivial(text), text
    for text in ["martha", "[PATIENT] Smith", "dax", "East Metro"]:
        assert not is_trivial(text), text


def test_teacher_dates_must_contain_a_calendar_element():
    from clinanon.distill.teacher import _CALENDAR

    for value in ["march", "Saturday", "August first", "this summer", "Christmas", "7:00", "2019", "the fall"]:
        assert _CALENDAR.search(value), value
    for value in ["last year", "next week", "weekly", "fall", "today", "this year", "6 months", "2 days ago"]:
        assert not _CALENDAR.search(value), value
