"""The teacher: an open-weight LLM served with vLLM (OpenAI-compatible API) on a rented GPU.

It marks private information inline with XML tags following our taxonomy, either in existing text
(``label_prompt``: copy the text and add tags) or in new encounters it writes (``generate_prompt``).
``parse_tagged`` turns tagged text back into plain text plus character spans; a labelled chunk is only
accepted if removing the tags gives back the input exactly.

    TEACHER_URL=http://localhost:8000/v1  (an SSH tunnel to the GPU box)
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

import httpx

TEACHER_URL = os.environ.get("TEACHER_URL", "http://localhost:8000/v1")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "teacher")

# Dates the teacher tags must contain a calendar element; relative time ("last week") is kept by policy.
_CALENDAR = re.compile(
    r"\b(?:19|20)\d\d\b|\b\d{1,2}[/.-]\d{1,2}\b|\b\d{1,2}:\d\d\b|\b\d{1,2}(?:st|nd|rd|th)\b"
    r"|\b\d{1,2} ?(?:am|pm|a\.m\.|p\.m\.)"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|march|april|may|june|july|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?|(?:mon|tues|wednes|thurs|fri|satur|sun)days?|christmas|thanksgiving|easter"
    r"|hanukkah|halloween|new year'?s?|birthday|memorial day|labor day|fourth of july|summer|winter|autumn"
    r"|(?:this|last|next|the) (?:fall|spring)|noon|midnight|o'clock)\b",
    re.IGNORECASE,
)

# Teacher tag -> taxonomy category (configs/privacy_taxonomy.yaml)
TAGS = {"NAME": "NAME", "AGE": "AGE", "DATE": "DATE", "TIME": "TIME", "LOCATION": "LOCATION",
        "ORGANIZATION": "ORGANIZATION", "PROFESSION": "PROFESSION", "CONTACT": "CONTACT", "ID": "IDENTIFIER"}
_TAG = re.compile(r"<(" + "|".join(TAGS) + r")>(.*?)</\1>")
_ANY_TAG = re.compile(r"</?(?:" + "|".join(TAGS) + r")>")

CATEGORIES = """\
<NAME> names of people: patient, relatives, friends, clinicians, anyone; also first names alone
<AGE> a person's age ("52-year-old", "i'm 44", "who is seventy-one", "aged 30")
<DATE> calendar dates and their parts: full dates, day and month, month names, years, weekdays, holidays, \
birthdays, named seasons ("this summer")
<TIME> clock times ("7:00", "2 am", "half past eight", "15.45")
<LOCATION> street addresses, postcodes, cities, regions, countries, and named places (parks, mountains, \
landmarks, neighbourhoods)
<ORGANIZATION> named hospitals, clinics, practices, employers, companies, stores, schools, sports teams, insurers
<PROFESSION> a person's job or occupation ("teacher", "bus driver", "software engineer")
<CONTACT> phone numbers, e-mail addresses, URLs, also when spoken ("zero seven nine ...", "anna dot keller at \
gmail dot com")
<ID> record numbers, insurance, policy, AHV or social security numbers, account or licence numbers"""

NEVER = """\
- medical terms, including eponyms (Parkinson's disease, Spurling's test, Lachman, Murphy's sign, Bell's palsy),
  diagnoses, symptoms, procedures, tests
- drug, product and device names (Lasix, Mobic, Camila, Aircast, Fitbit)
- relative time, durations and frequencies (today, yesterday, tonight, last week, two years ago, for 6 months,
  daily, twice a day)
- body parts, and clock-face positions in an exam ("a mass at the two o'clock position")
- generic places without a proper name (home, work, the gym, the beach, the emergency room, a trampoline park,
  the hand clinic, physical therapy)
- relationship words, roles and pronouns on their own (my wife, her son, dad, the nurse, my cardiologist, she)
- measurements, doses, vitals, lab values and scores (120/80, 7/10, 2/6 murmur, 50 mg)
- speaker labels such as [doctor] or [patient], and placeholders in square brackets such as [PATIENT]"""

LABEL_PROMPT = f"""You are annotating private information in clinical text for de-identification.

Copy the text between <<< and >>> exactly, character for character, and wrap every piece of private \
information in an XML tag that names its category. Change nothing else: keep the spelling, spacing, \
punctuation, line breaks and lower case exactly as they are.

Categories:
{CATEGORIES}

Never tag:
{NEVER}

Answer with the annotated text only, without the <<< and >>> markers.

<<<
%s
>>>"""

GENERATE_PROMPT = f"""Write one realistic medical encounter: the transcript of a conversation between a doctor and \
a patient, followed by the clinical note the doctor wrote about it.

Setting: {{setting}}. Main problem: {{problem}}.
The patient: {{persona}}.
The note: {{note_style}}.
Include, naturally and spread over the conversation and the note, these private details: {{details}}.
Also include, without tagging them: {{traps}}.

Format exactly like this:
DIALOGUE
[doctor] ...
[patient] ...
(25 to 50 turns; other speakers may appear as [patient_guest] or [nurse])
NOTE
... (section headers in capital letters on their own line)

Wrap every piece of private information, in both the conversation and the note, in an XML tag that names \
its category:
{CATEGORIES}

Never tag:
{NEVER}

All names, dates, numbers, addresses, places and organizations must be invented. Write nothing before \
DIALOGUE and nothing after the note."""


@dataclass(frozen=True)
class TaggedSpan:
    start: int
    end: int
    text: str
    category: str


def parse_tagged(tagged: str) -> tuple[str, list[TaggedSpan]]:
    """Plain text and the spans of the tagged values (nested or broken tags are removed, not kept)."""
    plain, spans, pos = [], [], 0
    length = 0
    for m in _TAG.finditer(tagged):
        before = _ANY_TAG.sub("", tagged[pos : m.start()])
        plain.append(before)
        length += len(before)
        value = _ANY_TAG.sub("", m.group(2))
        if value.strip():
            lead = len(value) - len(value.lstrip())
            stripped = value.strip()
            spans.append(TaggedSpan(length + lead, length + lead + len(stripped), stripped, TAGS[m.group(1)]))
        plain.append(value)
        length += len(value)
        pos = m.end()
    plain.append(_ANY_TAG.sub("", tagged[pos:]))
    return "".join(plain), spans


async def complete(client: httpx.AsyncClient, prompt: str, *, max_tokens: int, temperature: float = 0.0,
                   seed: int | None = None, thinking: bool = False) -> str:
    body = {
        "model": TEACHER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if temperature > 0:
        body.update(top_p=0.95)
    if seed is not None:
        body["seed"] = seed
    # A dropped tunnel or a restarting server must not kill a long run: retry for about two minutes, then
    # give up on this one request (the caller treats "" as a failed item).
    for attempt in range(8):
        try:
            r = await client.post(f"{TEACHER_URL}/chat/completions", json=body, timeout=1200)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, ValueError):
            await asyncio.sleep(min(30, 5 * (attempt + 1)))
    return ""


async def gather_limited(coros, limit: int = 64):
    semaphore = asyncio.Semaphore(limit)

    async def run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))
