"""Rule-based detectors: high-precision patterns for names in context, ages, dates, IDs and addresses.

Design notes
* Works on both cased notes and lower-cased ASR dialogues. Name tokens are validated differently:
  in cased text a name must be capitalized; in lower-cased text it must be a known first name/surname
  that is not an ordinary English word (otherwise "doctor said" would become a name).
* Clinical look-alikes are excluded by construction: dates need three parts or a month/year, so murmur
  grades ("2/6"), pain scores ("7/10") and blood pressures ("120/80") never match.
* Spans produced from unambiguous context (a title, a dictation header) are marked ``strong`` and are
  never overruled by the clinical veto ("mr. parkinson" stays a person).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import cache

from spacy.lang.en.stop_words import STOP_WORDS

from clinanon import resources
from clinanon.types import Span

# --------------------------------------------------------------------------------------------- helpers

# A name-like token: letters (any alphabet, so "Müller" works), inner apostrophes/hyphens allowed.
_NAME_TOKEN = r"[^\W\d_](?:[^\W\d_]|['\-](?=[^\W\d_]))*"
_SPELLED_NUMBER = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
)
_MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)
_MONTH_NOT_MAY = _MONTH.replace("may|", "")  # bare "may" is almost always the verb
_YEAR = r"(?:19|20)\d{2}"
_QUANTITY_UNIT = (r"(?:minutes?|seconds?|hours?|days?|weeks?|months?|years?|times|pounds?|percent|degrees|"
                  r"milligrams?|mg|units?|feet|miles|beats|breaths|reps|or|to)")
_UNITS_AFTER_NUMBER = (
    r"(?:mg|mcg|ml|cc|units?|calories|kcal|steps|ng|iu|feet|ft|miles|meters|lbs|pounds|kg|%|percent)\b"
)

_WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
_HOLIDAYS = (
    r"(?:christmas|thanksgiving|easter|halloween|new year'?s(?: eve| day)?|fourth of july|"
    r"independence day|memorial day|labor day|valentine'?s day|hanukkah|passover|ramadan|diwali)"
)
_RELATIVE_ANCHORS = r"(?:today|yesterday|tonight|tomorrow|last night|this (?:morning|afternoon|evening))"
_SMALL_NUMBER = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_ORDINAL_WORD = (
    r"(?:(?:twenty|thirty)[- ])?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
    r"twentieth|thirtieth"
)
# Years as spoken in ASR: "twenty eighteen", "nineteen ninety five", "twenty oh five", "two thousand ten".
_SPOKEN_YEAR = (
    r"(?:(?:nineteen|twenty)[- ](?:oh[- ](?:one|two|three|four|five|six|seven|eight|nine)|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
    r"|two thousand(?: and)?(?:[- ](?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[- ](?:one|two|three|four|five))?))?)"
)


def is_cased(text: str) -> bool:
    """True if the text uses normal capitalization (notes), False for lower-cased ASR (dialogues)."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.01


# Words that are never names here although some gazetteer lists them ("Doc", "Ay", "Patient" is a rare
# surname): forms of address, interjections and role nouns.
NON_NAME_WORDS = {
    "hey", "hi", "hello", "yes", "yeah", "yep", "no", "ay", "okay", "ok", "doc", "doctor", "dad", "daddy",
    "mom", "mommy", "sir", "ma'am", "maam", "man", "guys", "buddy", "honey", "dear", "folks", "everyone",
    "miss", "mister", "missus", "nurse", "babe", "son", "kiddo", "bud", "patient", "male", "female",
    "woman", "lady", "gentleman", "boy", "girl", "child", "baby", "infant", "pleasant", "alert", "today",
}  # fmt: skip


def is_frequent_name(token: str) -> bool:
    """Common enough as a name that a word like "mark" or "brown" can be trusted as one in context."""
    return resources.is_known_first_name(token, 1e-3) or resources.is_known_surname(token, 10_000)


def is_name_token(token: str, cased: bool, strict: bool = False, strong_context: bool = False) -> bool:
    """Could ``token`` be (part of) a person's name, given how the text is cased?

    ``strict`` additionally requires a known name in cased text (used when the context is weak).
    ``strong_context`` (a title, a dictation header, "X is a 59-year-old") accepts known names that are
    also ordinary or clinical words ("mister thompson", "so mark tell me").
    """
    low = token.lower()
    if len(low) < 2 or low in STOP_WORDS or low in NON_NAME_WORDS:
        return False
    known = resources.is_known_first_name(low, 1e-5) or resources.is_known_surname(low, 100)
    if cased:
        # ALL-CAPS words are section headers ("ASSESSMENT AND PLAN"), not names.
        return token[0].isupper() and not token.isupper() and (known or not strict)
    if len(low) < 3 or not known:
        return False
    is_word = low in resources.common_words() or low in resources.clinical_vocab()
    if strong_context:
        return not is_word or is_frequent_name(low)
    return not is_word


def is_name_continuation(token: str, cased: bool) -> bool:
    """Second/third token of a name already anchored by strong context ("eugene walker , date of birth").

    Ordinary words are allowed here because many surnames are words (Walker, Brown, Wood).
    """
    low = token.lower()
    if low in STOP_WORDS or low in NON_NAME_WORDS or len(low) < 2:
        return False
    if cased:
        return token[0].isupper() and not token.isupper()
    return low not in resources.clinical_vocab() or resources.is_known_surname(low, 1000)


def _span(m: re.Match, group: int | str, field: str, category: str, source: str, *,
          strong: bool = False, role: str | None = None) -> Span:
    return Span(field=field, start=m.start(group), end=m.end(group), text=m.group(group),
                category=category, source=f"rule:{source}", strong=strong, role=role)


# ----------------------------------------------------------------------------------------------- names

_TITLE_WITH_DOT = re.compile(
    rf"\b(?P<title>dr|mr|mrs|ms|mx|prof)\s*\.\s*(?P<n1>{_NAME_TOKEN})(?:\s+(?P<n2>{_NAME_TOKEN}))?", re.I)
_TITLE_NO_DOT = re.compile(
    rf"\b(?P<title>dr|mr|mrs|ms|mx|miss|mister|missus|doctor|nurse)\s+(?P<n1>{_NAME_TOKEN})"
    rf"(?:\s+(?P<n2>{_NAME_TOKEN}))?", re.I)
_CLINICIAN_TITLES = {"dr", "doctor", "prof", "nurse"}

# Dictation headers (virtscribe style): "next patient is randy gutierrez . date of birth , 8/10/2020"
_DICTATION = re.compile(
    rf"\b(?:next patient is|the patient is|patient is|patient name is|this is|dictating on|i am seeing|"
    rf"seeing)\s+(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})"
    rf"(?=\s*[,.]?\s*(?:date of birth|dob|mrn|medical record|,|\.))", re.I)
_DICTATION_TURN_START = re.compile(
    rf"^\[doctor\]\s+(?P<name>{_NAME_TOKEN}\s+{_NAME_TOKEN})\s*[,.]\s*(?:n-\s*)?(?:date of birth|dob)",
    re.I | re.M)
# "Andrew Campbell is a 59-year-old", "Ms. Philip Gutierrez is a pleasant 50-year-old",
# "Sophia Brown is a 75 y.o.". Name tokens are separated by spaces only (never a line break).
_SUBJECT_WITH_AGE = re.compile(
    rf"(?P<name>{_NAME_TOKEN}(?:[ \t]+{_NAME_TOKEN}){{0,2}})[ \t]+is\s+an?\s+(?:[a-z]+\s+){{0,2}}"
    rf"(?:\d{{1,3}}|{_SPELLED_NUMBER})\s*-?\s*(?:year|month|week|day|y\.?\s?o\b)", re.I)
_MY_NAME_IS = re.compile(
    rf"\bmy name is\s+(?P<title>(?:dr\.?|doctor)\s+)?(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?)", re.I)
# What may follow a name used to address someone: punctuation, end of turn, or the next clause.
_AFTER_ADDRESS = r"(?=\s*[,.!?]|\s*$|\s+(?:how|it's|nice|good|what|i|so|tell|can|let's|you|we|thanks)\b)"
# Greetings/vocatives: "hi , andrew .", "thanks , rachel", "hey there mark nice to see you"
_VOCATIVE = re.compile(
    rf"\b(?:hi|hello|hey|good morning|good afternoon|good evening|morning|thanks|thank you|bye|goodbye|"
    rf"welcome back|nice to meet you|nice to see you)\b(?:\s+there\b)?\s*,?\s*(?:miss\s+|mister\s+)?"
    rf"(?P<name>{_NAME_TOKEN}){_AFTER_ADDRESS}", re.I | re.M)
# Turn-initial address: "[doctor] so carolyn tell me about ...", "[doctor] alright okay theresa so ..."
_TURN_START_ADDRESS = re.compile(
    # \b after the filler: otherwise "so" matches the start of "sophia" and leaves "phia" as the name
    rf"^\[doctor\]\s+(?:(?:so|okay|ok|alright|well|now|and)\b\s*,?\s*){{1,3}}(?P<name>{_NAME_TOKEN})"
    rf"(?=\s*,?\s+(?:is|it's|tell|how|what|you|can|let's|i|so)\b)", re.I | re.M)


_RELATIVES = (r"(?:mom|mother|dad|father|wife|husband|son|daughter|sister|brother|aunt|uncle|niece|nephew|cousin|"
              r"grandmother|grandfather|grandma|grandpa|granddaughter|grandson|partner|boyfriend|girlfriend|"
              r"friend|neighbou?r|fianc[eé]e?|stepson|stepdaughter|parents?|kids?|children|child)")
# "my aunt , rembrandt , who ...", "my son jake is", "my mom, Aurélie, who is 61"
_RELATIVE_NAMED = re.compile(
    rf"\b(?:my|his|her|their|our|your)\s+{_RELATIVES}\s*,?\s+(?P<name>{_NAME_TOKEN})"
    rf"(?=\s*,|\s+(?:who|is|was|and|has|had|will|drove|came|helps|lives|brought|said|picked|called)\b)", re.I)
_NAME_APPOSITION = re.compile(
    rf"\b(?P<name>{_NAME_TOKEN})\s*,\s*(?:(?:\d{{1,2}}|{_SPELLED_NUMBER})\s*,(?=\s*(?:was|is|who|and|has|had|lives|drove|came|helps|said|will)\b)"
    rf"|who(?:'s|\s+is|\s+was)\s+(?:\d{{1,2}}|{_SPELLED_NUMBER})\b"
    rf"|(?:my|his|her|their|our)\s+(?:(?:\d{{1,2}}|{_SPELLED_NUMBER})[- ]year[- ]old\s+)?{_RELATIVES}\b)", re.I)
_LIVES_WITH = re.compile(rf"\b(?:lives|living|lived|resides|residing|stays|staying)\s+(?:alone\s+)?with\s+"
                         rf"(?P<name>{_NAME_TOKEN})", re.I)


# Words that end up next to a comma in speech ("um , my cousin said", "actually , my husband").
_FILLERS = {"um", "uh", "uhm", "umm", "er", "ah", "oh", "hmm", "mm", "mhm", "like", "well", "so", "yeah", "okay",
            "anyway", "now", "then", "too", "though", "again", "learned", "said", "says", "told"}


def _plausible_name(m: re.Match, text: str, cased: bool, need_known: bool) -> bool:
    """A name in apposition: not a filler or adverb; in cased text capitalized and (if ``need_known``) not
    capitalized just because it starts a sentence; in ASR a known name if ``need_known``."""
    name = m.group("name")
    low = name.lower()
    if low in _FILLERS or low.endswith("ly") or low in NON_NAME_WORDS or _is_plain_word(name):
        return False
    if cased:
        sentence_start = re.search(r"(?:^|[.!?:\]\n])\s*$", text[max(0, m.start("name") - 3) : m.start("name")])
        return name[0].isupper() and not (need_known and sentence_start)
    return not need_known or is_name_token(name, cased=False)


def _name_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    # "my aunt , rembrandt , who ..." is clear enough for unknown (European) names; a bare apposition
    # ("calogero , 35 , was") or "lives with X" needs a known name in ASR.
    for pattern, need_known in ((_RELATIVE_NAMED, False), (_LIVES_WITH, True), (_NAME_APPOSITION, True)):
        for m in pattern.finditer(text):
            if _plausible_name(m, text, cased, need_known):
                yield _span(m, "name", field, "NAME", "relative_named", strong=True)

    def tokens_ok(name: str, strict: bool, strong_context: bool = False) -> bool:
        """First token must look like a name; later tokens may be surname-words (Walker, Brown)."""
        first, *rest = name.split()
        return (is_name_token(first, cased, strict, strong_context)
                and all(is_name_continuation(t, cased) for t in rest))

    def addressed_name_ok(token: str) -> bool:
        """Name used to address someone: common first names are accepted even if they are words (Mark)."""
        return is_name_token(token, cased, strict=True) or (
            resources.is_known_first_name(token, 1e-4)
            and is_name_token(token, cased, strict=True, strong_context=True))

    for m in _TITLE_WITH_DOT.finditer(text):
        # "dr." / "mr ." are unambiguous titles: accept any following word that is not a stop word.
        if m.group("n1").lower() in STOP_WORDS:
            continue
        role = "clinician" if m.group("title").lower() in _CLINICIAN_TITLES else None
        end = m.end("n1")
        if m.group("n2") and is_name_token(m.group("n2"), cased, strict=True):
            end = m.end("n2")
        yield Span(field, m.start("n1"), end, text[m.start("n1"):end], "NAME", "rule:title",
                   strong=True, role=role)

    for m in _TITLE_NO_DOT.finditer(text):
        title = m.group("title").lower()
        # Without a dot the title must be followed by a known name ("doctor appointment" is not one).
        # "miss"/"nurse" are also ordinary words ("i miss work"), so their name must not be a word.
        strong = title not in {"miss", "nurse"}
        if not is_name_token(m.group("n1"), cased, strict=True, strong_context=strong):
            continue
        role = "clinician" if title in _CLINICIAN_TITLES else None
        end = m.end("n1")
        if m.group("n2") and is_name_token(m.group("n2"), cased, strict=True):
            end = m.end("n2")
        yield Span(field, m.start("n1"), end, text[m.start("n1"):end], "NAME", "rule:title",
                   strong=True, role=role)

    for pattern, source in ((_DICTATION, "dictation_header"), (_DICTATION_TURN_START, "dictation_header")):
        for m in pattern.finditer(text):
            if tokens_ok(m.group("name"), strict=True, strong_context=True):
                yield _span(m, "name", field, "NAME", source, strong=True, role="patient")

    for m in _SUBJECT_WITH_AGE.finditer(text):
        name = m.group("name")
        # Drop leading words that are not names ("so", "today", "mister") while keeping the name itself.
        tokens = name.split()
        while tokens and not is_name_token(tokens[0], cased, strict=True, strong_context=True):
            tokens = tokens[1:]
        if tokens and len(tokens) <= 3 and tokens_ok(" ".join(tokens), strict=True, strong_context=True):
            start = m.start("name") + name.index(tokens[0])
            yield Span(field, start, m.end("name"), text[start:m.end("name")], "NAME",
                       "rule:subject_with_age", strong=True, role="patient")

    for m in _MY_NAME_IS.finditer(text):
        if tokens_ok(m.group("name").split()[0], strict=False):
            role = "clinician" if m.group("title") else None
            end = m.end("name") if tokens_ok(m.group("name"), strict=True) else m.start("name") + len(
                m.group("name").split()[0])
            yield Span(field, m.start("name"), end, text[m.start("name"):end], "NAME", "rule:my_name_is",
                       strong=True, role=role)

    for pattern, source in ((_VOCATIVE, "vocative"), (_TURN_START_ADDRESS, "turn_start_address")):
        for m in pattern.finditer(text):
            if addressed_name_ok(m.group("name")):
                yield _span(m, "name", field, "NAME", source, strong=True)


# ------------------------------------------------------------------------------------------------ ages

_AGE_PATTERNS = [
    # "59-year-old", "3 -year-old", "18 months old", "eight years old"
    re.compile(rf"\b(?:\d{{1,3}}|{_SPELLED_NUMBER})\s*-?\s*(?:years?|yrs?|months?|weeks?|days?)\s*-?\s*old\b",
               re.I),
    re.compile(r"\b\d{1,3}\s*(?:years?|yrs?)\s+of\s+age\b", re.I),
    re.compile(r"\b\d{1,3}\s*(?:yo|y/o|y\.o\.)(?!\w)", re.I),
    re.compile(r"\bin (?:his|her|their) (?:early |mid |late |mid-)?(?:\d0s|twenties|thirties|forties|fifties|"
               r"sixties|seventies|eighties|nineties)\b", re.I),
]
# Only the number is the age: "aged 59", "age 45", "i'm 44 now"
_AGE_NUMBER_PATTERNS = [
    re.compile(r"\b(?:aged|age of|age)\s+(?P<age>\d{1,3})\b", re.I),
    # "i'm 44 now", "i was fourteen", "when you were fourteen", "my mom was almost 45 when ..."
    re.compile(rf"\b(?:i'm|i am|i was|you were|he's|he is|he was|she's|she is|she was|patient is|who's|who is|"
               rf"who was|who turned|who turns|"
               rf"my (?:mom|mother|dad|father|wife|husband|son|daughter|sister|brother|grandmother|grandfather|"
               rf"grandma|grandpa|aunt|uncle|brother|sister|son|daughter|cousin|nephew|niece|partner)"
               rf"(?:\s*,\s*{_NAME_TOKEN}\s*,)?\s+(?:is|was))"
               rf"\s+(?:(?:about|almost|nearly|only|like|just|around|indeed|already|now)\s+)?"
               rf"(?P<age>\d{{1,3}}|{_SPELLED_NUMBER})"
               # not a decimal or a quantity: "she is about 1.5 kg down", "about 3 , 3 and a half pounds"
               rf"(?![.,]\d)(?![^.\n]{{0,25}}\b(?:pounds?|lbs?|kg|kilos?|kilograms?|percent|points?|inches|"
               rf"feet|foot|cm|mm|degrees|out from|out of)\b)"
               # unpunctuated ASR runs straight into the next clause: "i was fourteen i was a gymnast";
               # "years" only as "years old" ("i'm eight years out from surgery" is not an age)
               rf"(?=\s*(?:years?\s+(?:old|of age)\b|now\b|and\b|when\b|at the time\b|i\b|i'm\b|so\b|"
               rf"but\b|then\b|recently\b|last\b|this\b|[,.!?]|$))",
               re.I | re.M),
]


# "his brother Ravi (30) has diabetes" (cased text only, see _age_spans)
_NAME_PAREN_AGE = re.compile(r"\b(?P<name>[A-ZÄÖÜÉÈ][^\W\d_]+)\s*\((?P<age>\d{1,2})\)")
# "calogero , 35 , was the one who found out": a name in apposition with an age.
_NAME_COMMA_AGE = re.compile(
    rf"\b(?P<name>{_NAME_TOKEN})\s*,\s*(?P<age>\d{{1,2}}|{_SPELLED_NUMBER})\s*,"
    rf"(?=\s*(?:was|is|who|and|has|had|lives|drove|came|helps|said|will)\b)", re.I)


def _is_plain_word(token: str) -> bool:
    low = token.lower()
    return low in STOP_WORDS or low in resources.common_words() or low in resources.clinical_vocab()


def _age_spans(text: str, field: str) -> Iterator[Span]:
    for m in _NAME_PAREN_AGE.finditer(text):
        if not _is_plain_word(m.group("name")):
            yield _span(m, "age", field, "AGE", "name_paren_age", strong=True)
    for m in _NAME_COMMA_AGE.finditer(text):
        if not _is_plain_word(m.group("name")) and m.group("name").lower() not in _FILLERS:
            yield _span(m, "age", field, "AGE", "name_comma_age", strong=True)
    for pattern in _AGE_PATTERNS:
        for m in pattern.finditer(text):
            yield _span(m, 0, field, "AGE", "age", strong=True)
    for pattern in _AGE_NUMBER_PATTERNS:
        for m in pattern.finditer(text):
            yield _span(m, "age", field, "AGE", "age_number", strong=True)


# ----------------------------------------------------------------------------------------------- dates

# Each pattern either matches the date itself or exposes it as the named group "d".
_DATE_PATTERNS = [
    ("numeric_date", re.compile(r"(?<![\d/.])\d{1,2}[/.\-]\d{1,2}[/.\-](?:\d{4}|\d{2})(?![\d/])")),
    ("iso_date", re.compile(rf"\b{_YEAR}-\d{{1,2}}-\d{{1,2}}\b")),
    ("day_mon_year", re.compile(rf"\b\d{{1,2}}[-/.]{_MONTH}[-/.](?:{_YEAR}|\d{{2}})\b", re.I)),
    ("month_year_numeric", re.compile(rf"(?<![\d/])(?:0?[1-9]|1[0-2])/{_YEAR}\b")),
    ("month_day_year", re.compile(
        rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:\s*,?\s*(?:{_YEAR}|'\d{{2}})\b)?", re.I)),
    # "14 May 1958", "the 5th of june"; bare "may" only with a year ("option 1 may be" is not a date)
    ("day_month_year", re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\.?\s+(?:of\s+)?(?:{_MONTH_NOT_MAY}\b\.?(?:\s*,?\s*{_YEAR}\b)?"
        rf"|may\s*,?\s*{_YEAR}\b)", re.I)),
    ("month_year", re.compile(rf"\b{_MONTH}\.?\s*(?:of\s+)?,?\s*{_YEAR}\b", re.I)),
    ("month_after_preposition", re.compile(
        rf"\b(?:in|of|since|until|till|by|during|early|mid|late|last|this|next|from|before|after|through)\s+"
        rf"(?:(?:about|around|approximately|roughly|early|mid|late|the end of|the beginning of)\s+)?"
        rf"(?P<d>{_MONTH_NOT_MAY})\b", re.I)),
    # Full month names alone ("november and december"); "march" and "may" are also verbs, so they need
    # the context of the patterns above, and abbreviations are clinical ("dec" = decreased, "oct").
    ("month_name", re.compile(
        r"\b(?:january|february|april|june|july|august|september|october|november|december)\b", re.I)),
    ("month_of_relative_year", re.compile(rf"\b{_MONTH}\s+of\s+(?:this|last|next)\s+year\b", re.I)),
    ("month_spelled_day", re.compile(rf"\b(?:{_MONTH_NOT_MAY}\s+(?:the\s+)?|may\s+(?!first\b))(?:{_ORDINAL_WORD})\b",
                                     re.I)),  # not "you may first try"
    # "the twenty-ninth of march", "the fifth of december nineteen eighty-nine" (the year matches separately)
    ("spelled_day_month", re.compile(rf"\b(?:the\s+)?(?:{_ORDINAL_WORD})\s+of\s+{_MONTH}\b", re.I)),
    ("year", re.compile(rf"(?<![\d/.\-<>]){_YEAR}(?![\d/])(?!\s*{_UNITS_AFTER_NUMBER})", re.I)),
    ("short_year", re.compile(r"(?<=\s)'\d{2}\b")),
    # Not "twenty twenty vision" (20/20) or "twenty thirty minutes" (a range of numbers).
    ("spoken_year", re.compile(rf"\b{_SPOKEN_YEAR}\b(?!\s+(?:vision|{_QUANTITY_UNIT})\b)", re.I)),
    # Day and month without a year. Pain scores ("4/10") and murmur grades ("2/6") look the same, so only
    # a zero-padded form ("04/10") or a date preposition ("on 4/10", "since 3/12") counts.
    ("zero_padded_day_month", re.compile(r"(?<![\d/.])0[1-9][/.\-](?:[0-2]\d|3[01])(?![\d/.])")),
    ("day_month_after_preposition", re.compile(
        r"\b(?:on|dated|since|until)\s+(?P<d>(?:0?[1-9]|1[0-2])[/\-](?:[0-2]?\d|3[01]))(?![\d/.\-])", re.I)),
    # ASR writes "10/10/2020" as "10-10 20-20".
    ("spoken_numeric_date", re.compile(r"\b\d{1,2}-\d{1,2} \d{2}-\d{2}\b")),
    # "scheduled on the 24th for surgery"; the lookahead keeps "the 5th digit" / "the 2nd time" out.
    ("ordinal_day", re.compile(
        rf"\b(?:on|by|until|till|before|after|since)\s+the\s+(?P<d>\d{{1,2}}(?:st|nd|rd|th))\b"
        rf"(?=\s*(?:[,.?!]|$|(?:for|at|and|or|when|is|so|i|we|you|then|of\s+{_MONTH})\b))", re.I | re.M)),
    ("weekday", re.compile(rf"\b{_WEEKDAYS}\b", re.I)),
    ("holiday", re.compile(rf"\b{_HOLIDAYS}\b", re.I)),
    # "fall" and "spring" are also an injury and a coil, so they need a determiner that makes them seasons.
    ("season", re.compile(
        r"\b(?:summer|winter|autumn|(?:this|last|next)\s+(?:fall|spring)|(?<=in the )spring)\b", re.I)),
]
_TIME_ANCHOR = re.compile(rf"\b{_RELATIVE_ANCHORS}\b", re.I)

_TIME_PATTERNS = [
    ("clock_time", re.compile(r"(?<![\d:/.])(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*(?:am|pm|a\.m\.|p\.m\.))?(?![\d:])",
                              re.I)),
    ("am_pm", re.compile(r"\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)(?!\w)", re.I)),  # not "which one am i"
    ("oclock", re.compile(rf"\b(?:\d{{1,2}}|{_SMALL_NUMBER}) o'clock\b", re.I)),
    ("spoken_time", re.compile(
        rf"\b(?:half|quarter)\s+(?:past|to)\s+{_SMALL_NUMBER}\b"
        rf"|(?<=\bat ){_SMALL_NUMBER}\s+(?:fifteen|thirty|forty[- ]five|o'?clock)\b", re.I)),
    # European "at 15.45"; not a dose or a measurement ("at 1.25 mg")
    ("dotted_time", re.compile(rf"(?<=\bat )(?:[01]?\d|2[0-3])\.[0-5]\d(?!\d|\.\d)(?!\s*{_UNITS_AFTER_NUMBER})", re.I)),
]
# Clock-face positions describe anatomy, not time: "a mass at the two o'clock position".
_CLOCK_POSITION_AFTER = re.compile(r"\s*(?:position|direction|axis|location|on the)\b", re.I)  # .match()
_CLOCK_POSITION_BEFORE = re.compile(r"\b(?:mass|lesion|nodule|tear|cyst|tumou?r|cm|mm|breast)\b[^.\n]{0,25}$",
                                    re.I)


def _date_spans(text: str, field: str) -> Iterator[Span]:
    for source, pattern in _DATE_PATTERNS:
        group = "d" if "d" in pattern.groupindex else 0
        for m in pattern.finditer(text):
            yield _span(m, group, field, "DATE", source, strong=True)
    for m in _TIME_ANCHOR.finditer(text):
        yield _span(m, 0, field, "TIME_ANCHOR", "time_anchor", strong=True)
    for source, pattern in _TIME_PATTERNS:
        for m in pattern.finditer(text):
            if _CLOCK_POSITION_AFTER.match(text, m.end()) or _CLOCK_POSITION_BEFORE.search(text, 0, m.start()):
                continue
            yield _span(m, 0, field, "TIME", source, strong=True)


# ------------------------------------------------------------------------------- identifiers & contact

_ID_PATTERNS = [
    # "mrn 3-8-4-9-2-0", "medical record number is 123546", "member id: AB12345"
    ("labeled_id", re.compile(
        r"\b(?:mrn|medical record(?: number)?|record(?: number| no)|chart number|patient (?:id|number)|pid|"
        r"account number|member (?:id|number)|insurance (?:id|number|card(?: number)?)|card number|policy|"
        r"id number|social security(?: number)?|ssn|ahv(?: number)?|avs)"
        # value: digits with inner separators; a dot/space only counts if another digit follows
        r"\b(?:\s*(?:number|no\.?|nr\.?|#|is|:|,|-))*\s*(?P<id>[a-z]{0,3}\d(?:[\d\-/]|[.\s](?=\d)){0,25})", re.I)),
    # Unlabeled numbers that are too long to be a measurement: "8075611338185", "6918 8981 8295",
    # "994.7458.564", "FW6410349" (blood sugars "140 150 160" have three digits, so spaces need 4-digit groups)
    ("long_number", re.compile(rf"(?<![\d.,/])\d{{7,}}(?!\d|[.,/]\d)(?!\s*{_UNITS_AFTER_NUMBER})", re.I)),
    ("grouped_number", re.compile(r"(?<![\d.,/])(?:\d{3,4}(?:[.\-]\d{3,4}){2,4}|\d{4}(?: \d{4}){2,3})(?!\d|[.,/]\d)")),
    ("alphanumeric_id", re.compile(r"\b[a-z]{1,3}-?\d{6,}\b", re.I)),
    ("glued_mrn", re.compile(r"\bmrn[a-z]?\d{3,}\b", re.I)),  # ASR glues label and value: "mrnr472348"
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("swiss_ahv", re.compile(r"\b756\.\d{4}\.\d{4}\.\d{2}\b")),
]
# Addresses are only matched in cased text: in lower-cased ASR "30 miles on a road" looks like one.
_ADDRESS_PATTERNS = [
    ("us_street", re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|"
        r"Dr|Court|Ct|Way|Place|Pl|Terrace|Circle)\b\.?")),
    ("eu_street", re.compile(
        r"\b[A-ZÄÖÜ][a-zäöüß]+(?:strasse|straße|gasse|weg|platz|allee|ring)\s+\d{1,4}[a-z]?\b")),
    # Postal code + city only with an explicit label: a bare "2019 He ..." would otherwise match.
    ("zip_with_city", re.compile(
        r"(?i:\b(?:zip(?: code)?|postal code|plz)\b)[:\s]*(?:CH-)?\d{4,5}(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?")),
]


_DIGIT_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_CONTACT_PATTERNS = [
    # "+41 79 123 45 67", "+41 (0)76 907 43 91", "+44(0)117 496 0081", "+39 0825604807"
    ("phone_international", re.compile(r"(?<![\w+])\+\d{1,3}[ \-]?(?:\(0\)[ \-]?)?\d{1,4}(?:[ \-]?\d{2,8}){1,4}(?!\d)")),
    # with a trunk zero: "079 123 45 67", "044 268 12 34", "01 55 66 90 74", "030 1234567"
    ("phone_national", re.compile(
        r"(?<![\d.,/:])0\d{1,3}(?:[ \-/]\d{2,4}(?:[ \-]\d{2,4}){1,3}|[ \-/]\d{6,8})(?!\d|[.,/:]\d)")),
    ("phone_area_code", re.compile(r"\(0\d{2,5}\)\s?\d{3,8}(?:[ \-]\d{2,4})?(?!\d)")),  # "(07243) 808264"
    # "(555) 123-4567", "555-123-4567", "001-874-539-1500x080"
    ("phone_us", re.compile(r"(?<![\d\-.])(?:(?:\+1|001)[\-. ])?(?:\(\d{3}\)\s?|\d{3}[\-.])\d{3}[\-.]\d{4}"
                            r"(?:\s*(?:x|ext\.?)\s*\d{1,5})?(?![\d\-.])", re.I)),
    # spoken digit by digit (ASR): "zero seven nine one two three four five six seven"
    ("phone_spoken", re.compile(rf"\b(?:{_DIGIT_WORD}[ ,\-]+){{6,}}{_DIGIT_WORD}\b", re.I)),
    ("email", re.compile(r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b")),
    # "anna dot keller at gmail dot com"
    ("email_spoken", re.compile(
        r"\b[a-z0-9]+(?:\s+(?:dot|underscore|dash)\s+[a-z0-9]+)*\s+at\s+[a-z0-9\-]+"
        r"(?:\s+dot\s+(?:com|net|org|edu|gov|ch|de|it|fr|uk|co|at|eu|info|io))+\b", re.I)),
]
# Streets in any casing: German compounds ("bahnhofstrasse 12") and Romance streets with a house number
# ("via roma 5", "rue boichat 78", "avenue François Bovet 38"); "rue"/"viale"/... are not English words,
# so they also count without a number ("rue Mélanie Monnet").
_STREET_LINK = r"(?:(?:de|des|du|d'|la|le|les|di|del|della|dei|delle|san|santa|saint|st\.?)\s+)*"
_ROMANCE_STREET = re.compile(
    rf"\b(?P<head>via|viale|corso|piazza|piazzale|vicolo|strada|rue|chemin|avenue|route|boulevard|quai|calle|rua)"
    rf"\s+{_STREET_LINK}(?P<name>{_NAME_TOKEN}(?:[ \t]+{_NAME_TOKEN}){{0,2}})"
    rf"(?P<number>\s*,?\s*\d{{1,4}}[a-z]?\b(?:,?\s+(?:piano|appartamento|apt\.?|apartment|suite|unit|flat)\s+\d{{1,4}})?)?",
    re.I)
_DISTINCTIVE_STREET_HEADS = {"viale", "corso", "piazza", "piazzale", "vicolo", "strada", "rue", "chemin", "quai",
                             "calle", "rua"}
_GERMAN_STREET = re.compile(r"\b[^\W\d_][\w\-]*?(?:strasse|straße|str\.|gasse|weg|platz|allee)\s*,?\s*\d{1,4}(?:[a-z]\b|-\d{1,4})?",
                            re.I)
_US_STATES = ("AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|"
              "ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY")
_POSTCODE_PATTERNS = [  # cased text only
    ("us_state_zip", re.compile(rf"\b(?:{_US_STATES})\s+\d{{5}}(?:-\d{{4}})?\b")),
    ("uk_postcode", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b(?:,?\s+[A-Z][a-z]+)?")),  # "LS7 6FH Brighton"
]


def _street_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    for m in _GERMAN_STREET.finditer(text):
        yield _span(m, 0, field, "LOCATION", "german_street", strong=True)
    for m in _ROMANCE_STREET.finditer(text):
        name = m.group("name").split()
        while name and _is_plain_word(name[-1]):
            name.pop()  # "rue boichat and ..." -> "rue boichat"
        if not name or all(_is_plain_word(t) for t in name):
            continue  # "via email", "route of administration"
        if cased and not all(t[0].isupper() for t in name):
            continue
        if m.group("number") and len(name) == len(m.group("name").split()):
            end = m.end("number")
        elif m.group("head").lower() in _DISTINCTIVE_STREET_HEADS:
            end = m.start("name") + len(" ".join(name))
        else:
            continue  # "avenue", "route" need a house number
        yield Span(field, m.start(), end, text[m.start() : end], "LOCATION", "rule:romance_street", strong=True)


# Labelled form fields in hospital documents ("PATIENT: SILVA, MARIA", "Age: 29", "MRN: CH-78421-99",
# "INSURER: HELSANA", "NURSE: J. VOGEL, RN"). The label decides the category; generic labels ("name",
# "physician") only count at the start of a line.
_FIELD_SEP = r"[ \t]*[:=][ \t]*"
_FORM_FIELDS = [
    ("NAME", re.compile(
        rf"(?:^[ \t•*\-]*(?:patient|name|physician|provider|therapist|pharmacist|nurse|clinician|gp)|"
        rf"\b(?:patient name|surname|first name|last name|next of kin|emergency contact|contact person|"
        rf"surgeon|assistants?|technologist|reported by|signed by|dictated by|transcribed by|"
        rf"(?:referring|attending|ordering|treating|primary care|family) (?:physician|doctor|provider)|"
        rf"anesthesi(?:ologist|a administered by)|pathologist|radiologist|interpreter))"
        rf"{_FIELD_SEP}(?P<value>[^\n|;(]+)", re.I | re.M)),
    ("AGE", re.compile(rf"\bage{_FIELD_SEP}(?P<value>\d{{1,3}}(?:[ \t]*(?:y|yrs?|years?(?: old)?|yo)\b)?)", re.I)),
    ("IDENTIFIER", re.compile(
        rf"\b(?:mrn|patient id|id number|(?:medical )?record (?:no|number)\.?|ins(?:urance)? (?:no|number|id)\.?|"
        rf"employee id|licen[cs]e (?:no|number)\.?|nhs number|case (?:no|number)\.?|accession (?:no|number)\.?|"
        rf"policy (?:no|number)\.?|member id|account (?:no|number)\.?|(?:visit|encounter|admission) (?:no|number|id)\.?)"
        rf"{_FIELD_SEP}(?P<value>(?=[^\n|;(]*\d)[A-Za-z0-9][\w\-./]*(?:[ ][\w\-./]*\d[\w\-./]*){{0,3}})", re.I)),
    ("ORGANIZATION", re.compile(
        rf"\b(?:insurer|insurance(?! (?:no|number|id))|hospital|clinic|service|facility|institution|employer|"
        rf"company|practice){_FIELD_SEP}(?P<value>[^\n|;(,]+)", re.I)),
    ("LOCATION", re.compile(rf"\b(?:address|home address|residence|place of birth|birthplace){_FIELD_SEP}"
                            rf"(?P<value>[^\n|;(]+)", re.I)),
]
_CREDENTIALS = re.compile(r"(?:,?\s*\b(?:md|do|rn|np|pa|phd|facs|ccrn|bsn|msn|mph|pharmd|dpt|lpn|cna)\b\.?)+\s*$", re.I)
_LEADING_TITLE = re.compile(r"^(?:(?:dr|mr|mrs|ms|prof)\.?\s+)+", re.I)
_NOT_A_VALUE = {"n/a", "na", "none", "unknown", "self", "same", "not applicable", "-", "--"}


def _form_field_spans(text: str, field: str) -> Iterator[Span]:
    for category, pattern in _FORM_FIELDS:
        for m in pattern.finditer(text):
            start, value = m.start("value"), m.group("value")
            if category in ("NAME", "ORGANIZATION", "LOCATION"):
                value = _CREDENTIALS.sub("", value.rstrip())
                lead = _LEADING_TITLE.match(value)
                if lead:
                    start, value = start + lead.end(), value[lead.end() :]
            value = value.rstrip(" \t.,")
            if not value.strip() or value.lower() in _NOT_A_VALUE:
                continue
            tokens = re.findall(r"[^\W\d_]+", value)
            if category == "NAME" and (re.search(r"\d", value) or len(tokens) > 6 or not tokens
                                       or all(_is_plain_word(t) for t in tokens)):
                continue  # "Patient: alert and oriented", "Physician: on call"
            if category == "ORGANIZATION" and all(_is_plain_word(t) for t in tokens):
                continue  # "Service: Internal Medicine"
            yield Span(field, start, start + len(value), value, category, f"rule:form_field_{category.lower()}",
                       strong=True)


def _id_and_address_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    yield from _form_field_spans(text, field)
    for source, pattern in _ID_PATTERNS:
        for m in pattern.finditer(text):
            group = "id" if "id" in pattern.groupindex else 0
            yield _span(m, group, field, "IDENTIFIER", source, strong=True)
    for source, pattern in _CONTACT_PATTERNS:
        for m in pattern.finditer(text):
            yield _span(m, 0, field, "CONTACT", source, strong=True)
    yield from _street_spans(text, field, cased)
    if cased:
        for source, pattern in _ADDRESS_PATTERNS + _POSTCODE_PATTERNS:
            for m in pattern.finditer(text):
                yield _span(m, 0, field, "LOCATION", source, strong=True)


# ------------------------------------------------------------------------------ places in any casing

_STREET_TYPE = r"(?:street|avenue|boulevard|lane|drive|road|st|ave|blvd)"
_NUMBERED_STREET_TYPE = r"(?:street|avenue|boulevard|lane|drive|road|court|terrace|circle|parkway|highway|st|ave|blvd)"
_DISTANCE_OR_TIME = (r"(?:miles?|blocks?|minutes?|hours?|seconds?|days?|weeks?|mph|km|kilometers?|meters?|"
                     r"feet|foot|yards?|steps|times)")
_ORDINAL_STREET = (r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
                   r"\d{1,3}(?:st|nd|rd|th))")
_PLACE_PATTERNS = [
    # "down on fifth street", "off main street", "123 oak street" (no stop words between number and type,
    # so "30 miles down the road" is not an address)
    ("street_name", re.compile(rf"\b(?:{_ORDINAL_STREET}\s+{_STREET_TYPE}|main\s+street)\b(?!\w)", re.I)),
    ("numbered_street", re.compile(
        rf"\b\d{{1,5}}\s+(?:(?!(?:the|a|an|on|down|up|of|in|to|at|and|{_DISTANCE_OR_TIME})\b)[a-z]+\s+){{1,2}}"
        rf"{_NUMBERED_STREET_TYPE}\b(?!\w)", re.I)),
    ("mountain", re.compile(rf"\bmount\s+(?:saint\s+|st\.?\s+)?(?P<name>{_NAME_TOKEN})", re.I)),
]
# Named facilities: "<name> hospital / medical / clinic / center / orthopedics". The word before the head
# must be a name, not a generic or clinical modifier ("the infusion center", "urgent care clinic").
# Bare "medical" only ends a name when nothing follows ("valley medical , and they ..."), otherwise
# "previous medical history" would match.
_FACILITY_HEAD = (r"(?:hospital|medical center|medical group|medical(?=\s*(?:[,.?!]|$|and\b|or\b|in\b|at\b|for\b|"
                  r"is\b|they\b))|clinic|health system|healthcare|orthopedics|surgical services|surgery center|"
                  r"center|centre|infirmary)")
_FACILITY = re.compile(rf"\b(?P<name>(?:{_NAME_TOKEN}[ \t]+){{1,3}}){_FACILITY_HEAD}\b", re.I | re.M)
# Types of place are kept: they are not identifying and ICD-10 place-of-occurrence codes (Y92) use them.
GENERIC_FACILITY_WORDS = {
    "general", "internal", "family", "primary", "urgent", "care", "emergency", "specialty", "medical",
    "local", "other", "same", "new", "old", "big", "small", "main", "nearest", "closest", "regular",
    "outpatient", "inpatient", "community", "rec", "recreation", "senior", "fitness", "health", "women's",
    "children's", "eye", "dental", "walk-in", "rehab", "rehabilitation", "infusion", "cancer", "heart",
    "pain", "wound", "sleep", "dialysis", "imaging", "surgery", "surgical", "sports", "medicine", "trauma",
    "treatment", "testing", "lab", "call", "shopping", "day", "student", "university", "breast", "free", "town",
    "orthopedic", "orthopedics", "spine", "vein", "weight", "loss", "diabetes", "kidney", "stroke",
    "physical", "therapy", "behavioral", "mental", "psychiatric", "addiction", "detox", "birthing", "va",
    "aquatic", "wreck", "civic", "convention", "youth", "activity", "wellness", "event", "swim", "swimming",
    "ice", "skating", "golf", "tennis", "bowling", "hand", "foot", "our", "their", "department",
}  # fmt: skip


def _is_facility_name_word(token: str) -> bool:
    low = token.lower()
    return not (low in STOP_WORDS or low in GENERIC_FACILITY_WORDS or low in resources.clinical_vocab())


def _is_proper_name_word(token: str, cased: bool) -> bool:
    return _is_facility_name_word(token) and (token[0].isupper() if cased else not _is_plain_word(token))


def _facility_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    for m in _FACILITY.finditer(text):
        # The name is the run of name words directly before the head: "the rex center" -> "rex center",
        # "we have a cancer center" -> nothing.
        words = list(re.finditer(r"\S+", m.group("name")))
        first = len(words)
        while first > 0 and _is_facility_name_word(words[first - 1].group()):
            first -= 1
        if first == len(words):
            # Generic words right before the head need a proper name before them: "Leeds General
            # Infirmary", "Massachusetts General Hospital", but not "consider hand clinic".
            generic = first
            while generic > 0 and words[generic - 1].group().lower() in GENERIC_FACILITY_WORDS:
                generic -= 1
            first = generic
            while first > 0 and _is_proper_name_word(words[first - 1].group(), cased):
                first -= 1
            if first == generic:
                first = len(words)
        if first < len(words):
            start = m.start("name") + words[first].start()
            yield Span(field, start, m.end(), text[start : m.end()], "ORGANIZATION", "rule:facility", strong=True)


# Cities, countries and regions from the GeoNames gazetteer, only after a word that introduces a place
# ("lives in winterthur", "moved here from bergamo", "a trip to lyon", "north of zurich"). Weak: the
# veto still sees the context ("Addison's disease"), and a longer overlapping span wins ("the Denver Nuggets").
_PLACE_CUE = re.compile(r"\b(?:in|from|to|near|outside(?:\s+of)?|around|of|visiting|visited)\s+(?:the\s+)?"
                        r"|(?P<postcode>\b(?:(?:CH|D|A|F|I|FL)-)?\d{4,5})\s+", re.I)
_PLACE_WORDS = re.compile(rf"{_NAME_TOKEN}\.?(?:[ \t]+{_NAME_TOKEN}\.?){{0,3}}")
_PLACE_CONNECTORS = {"am", "an", "im", "de", "di", "da", "del", "della", "der", "du", "la", "le", "les", "sur",
                     "upon", "on", "of"}  # "Frankfurt am Main", "Saint-Ouen-sur-Seine"


def _gazetteer_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    places = resources.places()
    for cue in _PLACE_CUE.finditer(text):
        m = _PLACE_WORDS.match(text, cue.end())
        if not m:
            continue
        words = list(re.finditer(r"\S+", m.group()))
        for k in range(len(words), 0, -1):  # longest name first: "new york city" before "new york"
            end = m.start() + words[k - 1].end()
            name = text[m.start() : end]
            if name.lower() not in places and name.endswith("."):
                name, end = name[:-1], end - 1  # sentence end: "in zurich."
            tokens = name.split()
            capitalized = tokens[0][0].isupper() and all(t[0].isupper() or t in _PLACE_CONNECTORS for t in tokens)
            postcode = cue.group("postcode")
            # After a postcode a capitalized word is a town even if it is not in the list ("1950 Sion"),
            # unless the number is a year ("2019 He was seen ...").
            unlisted_town = (postcode and cased and k == 1 and capitalized and not _is_plain_word(name)
                             and not re.fullmatch(r"(?:19|20)\d\d", postcode))
            if name.lower() not in places and not unlisted_town:
                continue
            if cased and not capitalized:
                break
            start = cue.start() if postcode else m.start()
            yield Span(field, start, end, text[start:end], "LOCATION", "gazetteer:geonames", strong=bool(postcode))
            break


# Hospitals named in German, French or Italian: "Kantonsspital Aarau", "Universitätsspital Zürich",
# "Ospedale Regionale di Lugano", "Hôpitaux Universitaires de Genève", "Klinik Hirslanden", "CHUV".
_FOREIGN_FACILITY = re.compile(
    r"\b(?:[\w\-]*?(?:klinik|klinikum|krankenhaus)|(?!hospitals?\b)[\w\-]*spital|ospedale|clinica|policlinico|h[oô]pital|"
    r"h[oô]pitaux|chuv|chu|charit[eé])\b", re.I)
_FACILITY_LINKS = {"de", "di", "du", "des", "del", "della", "dei", "la", "le", "am", "an", "der", "rechts", "links",
                   "universitaire", "universitaires", "universitario", "regionale", "cantonale", "civico", "san",
                   "santa", "st.", "st", "saint", "sankt"}


def _foreign_facility_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    for m in _FOREIGN_FACILITY.finditer(text):
        start, end = m.start(), m.end()
        # name words after the head (and one capitalized word before it: "Luzerner Kantonsspital")
        for w in re.compile(r"[ \t]+(st\.|\S+?)(?=[,.;:!?]?(?:\s|$))", re.I).finditer(text, end):
            if w.start() != end:
                break
            token = w.group(1)
            if token.lower() in _FACILITY_LINKS or (
                (token[0].isupper() if cased else not _is_plain_word(token)) and token.lower() not in STOP_WORDS
            ):
                end = w.end(1)
            else:
                break
        while text[start:end].split()[-1].lower() in _FACILITY_LINKS and end > m.end():
            end = start + text[start:end].rstrip().rfind(" ")  # never end on "di" / "de"
        before = re.search(r"(\S+)[ \t]+$", text[max(0, start - 30) : start])
        if before and not _is_plain_word(before.group(1)) and (not cased or before.group(1)[0].isupper()):
            start -= len(before.group(0))
        yield Span(field, start, end, text[start:end], "ORGANIZATION", "rule:foreign_facility", strong=True)


# Employers and insurers named after a cue: "i work at migros", "works as a nurse for White LLC",
# "insured with Helsana", "Insurance: CSS". The name is the run of capitalized words (cased text) or of
# words that are not ordinary or clinical (ASR).
_ORG_CUE = re.compile(
    r"\b(?:(?:i|he|she|they|we|you|who|still|also|currently|now|and|to|will|would|while|when|but|then|just|"
    r"i'll|ever|both|patient|husband|wife|son|daughter|mom|dad|mother|father|brother|sister|friend)\s+"
    r"(?:work(?:s|ed)?|working)(?:\s+(?:part[- ]time|full[- ]time))?(?:\s+as\s+an?\s+[\w\- ]{2,40}?)?"
    r"\s+(?:for|at|with)|employed\s+(?:as\s+an?\s+[\w\- ]{2,40}?\s+)?(?:by|at|with|for)|employer(?:\s+is|\s*:)"
    r"|(?:job|position|career)\s+(?:at|with)|insured\s+(?:with|by|at|through)"
    r"|(?:health\s+)?insur(?:ance|er)(?:\s+company|\s+provider)?(?:\s+is(?:\s+with)?|\s+through|\s*:)"
    r"|covered\s+by|company\s+called)\s+(?:the\s+)?", re.I)
_ORG_SUFFIX = {"ag", "gmbh", "sa", "sarl", "srl", "spa", "llc", "inc", "inc.", "ltd", "ltd.", "plc", "kg", "co.",
               "&", "cie.", "e.v."}


def _org_cue_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    for cue in _ORG_CUE.finditer(text):
        if _NOT_A_JOB_SUBJECT.search(text, max(0, cue.start() - 20), cue.start()):
            continue  # "the medication is working for me"
        end, pos, n = cue.end(), cue.end(), 0
        for w in re.compile(r"(\S+?)(?=[,.;:!?]?(?:\s|$))[ \t]*").finditer(text, cue.end()):
            if w.start() != pos or n == 4:
                break
            pos = w.end()
            token = w.group(1)
            low = token.lower()
            if not any(c.isalnum() for c in token) and token != "&":
                break  # punctuation ends the name ("for the sbb , my daughter")
            if n and low in _ORG_SUFFIX:
                ok = True
            elif cased:
                ok = token[0].isupper() and low not in STOP_WORDS
            else:
                ok = low not in STOP_WORDS and not low.endswith("ing") and not (n == 0 and _is_plain_word(token))
            if not ok:
                break
            end, n = w.start() + len(token), n + 1
            if text[end : end + 1] in ",.;:!?":
                break
        if n:
            yield Span(field, cue.end(), end, text[cue.end() : end], "ORGANIZATION", "rule:org_cue", strong=True)


def _place_spans(text: str, field: str, cased: bool) -> Iterator[Span]:
    for source, pattern in _PLACE_PATTERNS:
        for m in pattern.finditer(text):
            if source == "mountain" and m.group("name").lower() in STOP_WORDS:
                continue  # "mount a response"
            yield _span(m, 0, field, "LOCATION", source, strong=True)
    yield from _facility_spans(text, field, cased)
    yield from _foreign_facility_spans(text, field, cased)
    yield from _org_cue_spans(text, field, cased)
    yield from _gazetteer_spans(text, field, cased)


# ------------------------------------------------------------------------------- professions & vendors

# "she is a teacher", "works as a nurse", "retired police officer". After a copula the article is required,
# so a clinician's introduction ("i'm doctor smith") is not read as an occupation.
_COPULA_ARTICLE = (r"(?:(?:i'm|i am|i was|he's|he is|he was|she's|she is|she was|they're|you're|you are|is|was)"
                   r"\s+an?|(?:works?|worked|working|employed|job|career)\s+as(?:\s+an?)?|retired)"
                   r"\s+(?:retired\s+|former\s+|part-time\s+|full-time\s+)?")
_OWNS_BUSINESS = re.compile(
    r"\b(?:i|he|she|they|we|patient)\s+(?:own|owns|run|runs|owned|ran)\s+(?:an?|the|that|their|his|her|my)\s+"
    r"(?:that\s+|the\s+)*(?P<job>(?:[a-z\-]+\s+){0,2}(?:shop|store|business|company|restaurant|farm|bakery|salon|garage|bar|"
    r"cafe|firm|practice|daycare))\b", re.I)
# Ambient documentation assistants addressed by name ("hey dragon", "i'd like to tell dax") and the patient
# portal ("send me a mychart message"). They reveal the care setting's vendor (VENDOR in the Stanford PHI
# scheme) and PII detectors read them as persons. "my chart" in two words is the doctor's chart.
_VENDOR = re.compile(r"\b(?:dragon|dax)\b(?!\s+fruit)|\b(?:epic\s+)?mychart\b", re.I)


@cache
def _occupation_pattern() -> re.Pattern:
    jobs = sorted(resources.occupations(), key=len, reverse=True)  # longest first: "truck driver"
    alternatives = "|".join(re.escape(j).replace(r"\ ", r"[\s\-]+") for j in jobs)
    return re.compile(rf"\b{_COPULA_ARTICLE}(?P<job>(?:[a-z\-]+\s+){{0,2}}?(?:{alternatives})s?)\b(?!['’]s)",
                      re.I)  # not "a doctor's office"


# Any job after an explicit cue: "works as a dental assistant at", "Occupation: bank clerk".
_JOB_CUE = re.compile(r"\b(?:works?|worked|working|employed|job|career|position)(?:\s+(?:part[- ]time|full[- ]time))?"
                      r"\s+as\s+(?:an?\s+)?"
                      r"|\b(?:profession|occupation)(?:\s+is|\s*:)\s*(?:an?\s+)?|\bby profession\s+", re.I)
_NOT_A_JOB_SUBJECT = re.compile(r"\b(?:it|this|that|which|what|medication|medicine|drug|pill|treatment)\s+$", re.I)


def _profession_spans(text: str, field: str) -> Iterator[Span]:
    # Weak: the veto keeps occupations that are part of a diagnosis ("farmer's lung", J67.0).
    for m in _occupation_pattern().finditer(text):
        yield _span(m, "job", field, "PROFESSION", "occupation")
    for cue in _JOB_CUE.finditer(text):
        if _NOT_A_JOB_SUBJECT.search(text, max(0, cue.start() - 20), cue.start()):
            continue  # "it works as a blood thinner"
        words = []
        for w in re.compile(r"([a-zA-Z][a-zA-Z\-]*)[ \t]*").finditer(text, cue.end()):
            if (words and w.start() != words[-1].end()) or w.group(1).lower() in STOP_WORDS or len(words) == 3:
                break
            words.append(w)
        if words and not (len(words) == 1 and words[0].group(1).lower() in resources.clinical_vocab()):
            yield Span(field, words[0].start(), words[-1].end(1), text[words[0].start() : words[-1].end(1)],
                       "PROFESSION", "rule:job_cue")
    for m in _OWNS_BUSINESS.finditer(text):
        yield _span(m, "job", field, "PROFESSION", "owns_business")
    for m in _VENDOR.finditer(text):
        yield _span(m, 0, field, "VENDOR", "assistant_name", strong=True)


# ------------------------------------------------------------------------------------------- interface


def detect(text: str, field: str) -> list[Span]:
    """All rule-based candidate spans in one field of one encounter."""
    cased = is_cased(text)
    spans: list[Span] = []
    spans.extend(_name_spans(text, field, cased))
    spans.extend(_age_spans(text, field))
    spans.extend(_date_spans(text, field))
    spans.extend(_id_and_address_spans(text, field, cased))
    spans.extend(_place_spans(text, field, cased))
    spans.extend(_profession_spans(text, field))
    return spans
