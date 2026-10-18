"""Training data for the student, made by the teacher LLM (see src/clinanon/distill/teacher.py).

    label     the teacher tags private information in real encounters (the 140 held-out ACI encounters;
              the 67 eval encounters stay out of training for the go/no-go check)
    generate  the teacher writes new encounters with every private detail tagged inline: many specialties,
              note formats and patient origins; half are converted to the lower-cased ASR style of ACI
    robust    held-out robustness set: document and conversation types never used for training
              (radiology, pathology, ICU nursing, operative reports, portal messages, phone triage, ...),
              with ICD-10-CM codes chosen by the teacher, labelled in a separate pass

    TEACHER_URL=http://localhost:8000/v1 uv run python scripts/teacher_data.py label
    TEACHER_URL=http://localhost:8000/v1 uv run python scripts/teacher_data.py generate --n 1200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import httpx

from clinanon import resources
from clinanon.distill.teacher import (
    _ANY_TAG,
    _CALENDAR,
    GENERATE_PROMPT,
    LABEL_PROMPT,
    complete,
    gather_limited,
    parse_tagged,
)
from clinanon.evaluation.judges import chunks
from clinanon.evaluation.metrics import is_trivial
from clinanon.io import read_jsonl, write_jsonl
from clinanon.pipeline import normalize_asr_splits
from clinanon.veto import _CLINICAL_HEAD_AFTER

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "distill"
_TIME_LIKE = re.compile(r"\d|o'clock|half past|quarter|noon|midnight", re.I)


_TITLE = re.compile(r"^(?:dr|mr|mrs|ms|miss|mister|doctor|prof)\s*\.?\s+", re.I)


def trim_title(span: dict) -> dict:
    """Our convention tags the name, not the title: "dr. reyes" -> "reyes"."""
    m = _TITLE.match(span["text"]) if span["category"] == "NAME" else None
    if not m or m.end() == len(span["text"]):
        return span
    return {**span, "start": span["start"] + m.end(), "text": span["text"][m.end() :]}


_SPOKEN_OR_WRITTEN_CONTACT = re.compile(r"@|\bat\b.*\bdot\b|\d{3}.*\d{2}")


def clinical_mistake(span: dict, text: str) -> str | None:
    """Teacher labels that our taxonomy keeps: eponyms, drugs, clinical vocabulary (reason, or None)."""
    value = span["text"]
    tokens = [t.removesuffix("'s").removesuffix("’s") for t in re.findall(r"[a-zà-ÿ'’]+", value.lower())]
    after = text[span["end"] : span["end"] + 40]
    if span["category"] == "NAME":
        if re.search(r"['’]s$", value) and (_CLINICAL_HEAD_AFTER.match(after) or tokens[-1] in resources.eponyms()):
            return "eponym"
        if _CLINICAL_HEAD_AFTER.match(after) and not re.match(r"\s*['’]s\s+(?:test|exam)", after):
            return "eponym before a clinical head noun"
    if tokens and all(t in resources.drug_tokens() or t in resources.clinical_vocab() for t in tokens) and not (
        span["category"] == "NAME" and any(resources.is_known_first_name(t, 1e-4) for t in tokens)):
        return "clinical vocabulary"
    return None


def fix_category(span: dict) -> dict:
    if span["category"] == "NAME" and _SPOKEN_OR_WRITTEN_CONTACT.search(span["text"]):
        return {**span, "category": "CONTACT"}
    return span


def keep_span(category: str, text: str) -> bool:
    """Drop what the taxonomy keeps even if the teacher tagged it (relative time, pronouns, tags)."""
    if is_trivial(text):
        return False
    if category == "DATE" and not _CALENDAR.search(text):
        return False  # "last week", "today"
    return category != "TIME" or bool(_TIME_LIKE.search(text))


def _relocate(spans, source: str) -> list:
    """Spans of a chunk the teacher did not copy exactly: find each value in the source, in order."""
    found, pos = [], 0
    for s in spans:
        i = source.find(s.text, pos)
        if i < 0:
            i = source.find(s.text)
        if i >= 0:
            found.append((i, i + len(s.text), s.text, s.category))
            pos = i + len(s.text)
    return found


# ---------------------------------------------------------------------------------------------- label


async def label_text(client: httpx.AsyncClient, text: str, stats: Counter) -> list[dict]:
    spans = []
    pieces = list(chunks(text, max_words=220))

    async def one(offset: int, chunk: str):
        answer = await complete(client, LABEL_PROMPT % chunk, max_tokens=2 * len(chunk) // 3 + 400)
        answer = answer.strip().removeprefix("<<<").removesuffix(">>>").strip("\n")
        plain, found = parse_tagged(answer)
        if plain.strip() == chunk.strip():
            shift = chunk.find(plain.strip()) - plain.find(plain.strip())
            stats["chunks_exact"] += 1
            return [(offset + s.start + shift, offset + s.end + shift, s.text, s.category) for s in found]
        stats["chunks_relocated"] += 1
        return [(offset + a, offset + b, t, c) for a, b, t, c in _relocate(found, chunk)]

    for result in await asyncio.gather(*(one(o, c) for o, c in pieces)):
        for start, end, value, category in result:
            assert text[start:end] == value, (text[start:end], value)
            span = fix_category(trim_title({"start": start, "end": end, "text": value, "category": category}))
            if not keep_span(span["category"], value):
                stats["dropped_by_policy"] += 1
            elif reason := clinical_mistake(span, text):
                stats[f"dropped_{reason}"] += 1
            else:
                spans.append(span)
    return spans


async def label(args) -> None:
    records = read_jsonl(args.input)[: args.limit or None]
    stats: Counter = Counter()
    limits = httpx.Limits(max_connections=4 * args.concurrency, max_keepalive_connections=4 * args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:

        async def one(r):
            out = {"encounter_id": r["encounter_id"], "dataset": r["dataset"], "icd10_codes": r["icd10_codes"],
                   "spans": []}
            for field in ("dialogue", "note"):
                text = normalize_asr_splits(r[field])
                out[field] = text
                out["spans"] += [{**s, "field": field} for s in await label_text(client, text, stats)]
            return out

        labelled = await gather_limited([one(r) for r in records], limit=args.concurrency)
    write_jsonl(args.output, labelled)
    by_cat = Counter(s["category"] for r in labelled for s in r["spans"])
    print(f"{len(labelled)} encounters -> {args.output}\n{dict(stats)}\n{dict(by_cat)}")


# ------------------------------------------------------------------------------------------- generate

SETTINGS = [
    "family medicine visit", "orthopedic clinic", "emergency department", "cardiology follow-up",
    "dermatology clinic", "pediatric visit with a parent present", "geriatric consultation with a relative",
    "hospital discharge conversation", "telehealth video visit", "gynecology visit", "psychiatry follow-up",
    "oncology consultation", "neurology clinic", "endocrinology follow-up", "sports medicine clinic",
    "occupational health visit", "urology clinic", "gastroenterology clinic", "pulmonology clinic",
    "rheumatology clinic", "ENT clinic", "nephrology follow-up", "home visit by the family doctor",
    "walk-in urgent care", "pre-operative assessment",
]
ORIGINS = [
    ("a Swiss German", "Zurich, Bern, Basel, Lucerne, St. Gallen or Winterthur"),
    ("a Swiss French", "Geneva, Lausanne, Fribourg, Neuchâtel or Sion"),
    ("an Italian-speaking Swiss", "Lugano, Bellinzona, Locarno or Mendrisio"),
    ("a German", "Munich, Stuttgart, Freiburg or Cologne"), ("an Austrian", "Vienna, Graz or Innsbruck"),
    ("an Italian", "Milan, Como, Turin or Bergamo"), ("a French", "Lyon, Annecy or Strasbourg"),
    ("a British", "Manchester, Leeds, Bristol or Edinburgh"), ("an American", "a US city of your choice"),
    ("a Portuguese", "Switzerland"), ("a Kosovar Albanian", "Switzerland"), ("a Turkish", "Germany"),
    ("a Sri Lankan Tamil", "Switzerland"), ("a Spanish", "Switzerland"), ("a Polish", "Germany"),
    ("an Eritrean", "Switzerland"), ("a Serbian", "Austria"), ("a Brazilian", "Italy"),
]
NOTE_STYLES = [
    "sections CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, REVIEW OF SYSTEMS, PHYSICAL EXAMINATION, RESULTS, "
    "ASSESSMENT AND PLAN",
    "a SOAP note (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN)",
    "a hospital discharge summary with an administrative header (patient name, date of birth, record number)",
    "a referral letter to a specialist, starting with 'Dear Dr. ...'",
    "an emergency department note with times of arrival and discharge",
    "a short progress note",
]
DETAILS = [
    "the patient's full name", "the patient's date of birth", "the patient's age", "a phone number",
    "an e-mail address", "the home address with postcode and town", "the town or country they moved from",
    "the health insurer and the insurance number", "a medical record number", "the patient's job and employer",
    "the first names and ages of relatives", "the name of the referring doctor and their hospital or practice",
    "the dates of earlier visits, tests or operations", "the date and time of the next appointment",
    "a hobby tied to a named club, team, gym or place", "a recent trip to a named city or country",
    "the name of a colleague, friend or neighbour",
    "a person whose surname is also the name of a disease mentioned in the visit (tag the person, never the disease)",
]
TRAPS = [
    "medical eponyms such as Parkinson's, Crohn's, Hashimoto's, Graves' disease, Bell's palsy or Baker's cyst",
    "relative time such as yesterday, two weeks ago, for three months",
    "drug brand names", "vital signs, lab values and pain scores",
    "family history with relationship words (my mother, his brother)",
    "the type of place where an injury happened (a playground, the gym, a ski slope, the kitchen)",
    "an exam finding at a clock-face position (at the 3 o'clock position)",
]
SPEECH = ["natural speech with fillers (um, uh, you know), interruptions and repetitions",
          "a clean, edited transcript"]


def generation_config(rng: random.Random, problems: list[str]) -> dict:
    origin, place = rng.choice(ORIGINS)
    density = rng.random()
    n_details = rng.randint(2, 4) if density < 0.3 else rng.randint(5, 9)
    return {
        "setting": rng.choice(SETTINGS),
        "problem": rng.choice(problems),
        "persona": f"{origin} {rng.choice(['woman', 'man'])} aged {rng.randint(18, 92)}, living in {place}; "
                   f"conversation style: {rng.choice(SPEECH)}",
        "note_style": rng.choice(NOTE_STYLES),
        "details": "; ".join(rng.sample(DETAILS, n_details)),
        "traps": "; ".join(rng.sample(TRAPS, 3)),
    }


def _asr(text: str) -> str:
    text = re.sub(r"\s*([,.?!])(?=\s|$)", r" \1", text.lower())
    return re.sub(r"[;:\"“”]", "", text)


def to_asr(text: str, spans: list[dict]) -> tuple[str, list[dict]]:
    """ACI ASR style (lower case, spaced punctuation) with the spans moved along."""
    out, new_spans, pos = "", [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out += _asr(text[pos : s["start"]])
        value = text[s["start"] : s["end"]].lower()
        new_spans.append({**s, "start": len(out), "end": len(out) + len(value), "text": value})
        out += value
        pos = s["end"]
    return out + _asr(text[pos:]), new_spans


def split_encounter(plain: str, spans) -> dict | None:
    m = re.search(r"^\s*DIALOGUE\s*\n(.*?)\n\s*NOTE\s*\n(.*)$", plain, flags=re.S | re.M)
    if not m:
        return None
    fields = {"dialogue": (m.start(1), m.end(1)), "note": (m.start(2), m.end(2))}
    out = {"dialogue": m.group(1).strip("\n"), "note": m.group(2).strip()}
    if len(re.findall(r"^\[[a-z_]+\]", out["dialogue"], flags=re.M)) < 10 or len(out["note"]) < 300:
        return None
    out["spans"] = []
    for s in spans:
        for field, (a, b) in fields.items():
            if a <= s.start and s.end <= b:
                shift = a + (len(plain[a:b]) - len(plain[a:b].lstrip("\n")))
                start = s.start - shift
                if out[field][start : start + len(s.text)] == s.text and keep_span(s.category, s.text):
                    out["spans"].append(trim_title({"field": field, "start": start, "end": start + len(s.text),
                                                    "text": s.text, "category": s.category}))
    return out


async def generate(args) -> None:
    problems = sorted({c["description"] for f in ("eval_patient_doctor_encounters", "patient_doctor_encounters")
                       for r in read_jsonl(ROOT / "data" / "raw" / f"{f}.jsonl") for c in r["icd10_codes"]})
    rng = random.Random(args.seed)
    configs = [generation_config(rng, problems) for _ in range(args.n)]
    stats: Counter = Counter()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sink = open(args.output, "w", encoding="utf-8")  # noqa: SIM115 - written as results arrive
    limits = httpx.Limits(max_connections=args.concurrency + 16, max_keepalive_connections=args.concurrency + 16)
    async with httpx.AsyncClient(limits=limits) as client:

        async def one(i: int, config: dict):
            answer = await complete(client, GENERATE_PROMPT.format(**config), max_tokens=7000, temperature=0.8,
                                    seed=args.seed * 100_000 + i)
            plain, spans = parse_tagged(answer)
            encounter = split_encounter(plain, spans)
            if not encounter or len(encounter["spans"]) < 3:
                stats["rejected"] += 1
                return None
            # The tags written while generating mark eponyms and drugs too often; a separate labelling
            # pass over the plain text (the same as for real encounters) is much more precise.
            written = {(s["field"], s["start"], s["end"]) for s in encounter["spans"]}
            encounter["spans"] = []
            for field in ("dialogue", "note"):
                labelled = await label_text(client, encounter[field], stats)
                encounter["spans"] += [{**s, "field": field} for s in labelled]
            labelled = {(s["field"], s["start"], s["end"]) for s in encounter["spans"]}
            stats["spans_written"] += len(written)
            stats["spans_labelled"] += len(labelled)
            stats["spans_agree"] += len(written & labelled)
            asr = rng.random() < 0.5
            record = {"encounter_id": f"TEACHER-{args.seed}-{i:05d}", "dataset": "teacher-asr" if asr else "teacher-cased",
                      "config": config}
            spans_by_field = {f: [s for s in encounter["spans"] if s["field"] == f] for f in ("dialogue", "note")}
            record["note"] = encounter["note"]
            record["dialogue"], spans_by_field["dialogue"] = (to_asr(encounter["dialogue"], spans_by_field["dialogue"])
                                                              if asr else (encounter["dialogue"], spans_by_field["dialogue"]))
            record["spans"] = spans_by_field["dialogue"] + spans_by_field["note"]
            for s in record["spans"]:
                assert record[s["field"]][s["start"] : s["end"]] == s["text"]
            stats["accepted"] += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            return record

        results = await gather_limited([one(i, c) for i, c in enumerate(configs)], limit=args.concurrency)
    sink.close()
    records = [r for r in results if r]
    by_cat = Counter(s["category"] for r in records for s in r["spans"])
    print(f"{len(records)} encounters -> {args.output}\n{dict(stats)}\n{dict(by_cat)}")


# ------------------------------------------------------------------------------------------- robust

DOCUMENTS = [
    ("a hospital radiology department", "a radiology report (CT, MRI or X-ray) with an administrative header, "
     "clinical indication, technique, findings and impression"),
    ("a pathology institute", "a pathology report with specimen, macroscopy, microscopy and diagnosis"),
    ("an intensive care unit", "an ICU nursing progress note for one shift"),
    ("a surgical department", "an operative report: pre- and post-operative diagnosis, procedure, surgeon and "
     "assistants, anesthesia, findings and course of the operation"),
    ("a physiotherapy practice", "a physiotherapy progress report to the referring doctor"),
    ("a psychiatric clinic", "a psychiatric admission note with a mental status examination"),
    ("a home care nursing service", "a home care nursing visit report"),
    ("an occupational medicine service", "a fitness-for-work assessment addressed to the employer"),
    ("a patient portal", "a thread of secure portal messages between the patient and the practice"),
    ("a hospital pharmacy", "a medication reconciliation note at admission"),
]
CONVERSATIONS = [
    ("a nurse phone triage call", "[nurse], [patient]"),
    ("a nursing shift handover at the bedside", "[nurse], [nurse_2], [patient]"),
    ("a ward round with the senior doctor, a resident, the patient and a relative",
     "[doctor], [resident], [patient], [patient_guest]"),
    ("a pharmacist counselling the patient about a new medication", "[pharmacist], [patient]"),
    ("a paramedic handover to the emergency team", "[paramedic], [doctor], [patient]"),
    ("a consultation through a professional interpreter", "[doctor], [interpreter], [patient]"),
]
ROBUST_PROMPT = """Write one realistic set of clinical documentation from {setting}, in two parts.

Part 1, CONVERSATION: {conversation}. Each turn on its own line, starting with the speaker in square \
brackets in lower case ({speakers}). 15 to 40 turns.
Part 2, DOCUMENT: {document}. Write it exactly as such a document looks (header lines, sections in capital \
letters where they belong).

The patient: {persona}. Choose a medical problem that fits this setting.
Include naturally, spread over both parts, these private details: {details}.
All names, dates, numbers, addresses, places and organizations must be invented.
Finally list the ICD-10-CM codes of the documented diagnoses.

Format exactly like this, with no markup or tags:
CONVERSATION
...
DOCUMENT
...
ICD10
<code> | <description>
(one line per code, 1 to 4 codes)"""
_ICD_LINE = re.compile(r"^\s*([A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?)\s*\|\s*(.+?)\s*$", re.M)


async def robust(args) -> None:
    rng = random.Random(args.seed)
    stats: Counter = Counter()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    records = []
    limits = httpx.Limits(max_connections=args.concurrency + 16, max_keepalive_connections=args.concurrency + 16)
    async with httpx.AsyncClient(limits=limits) as client:

        async def one(i: int):
            setting, document = DOCUMENTS[i % len(DOCUMENTS)]
            conversation, speakers = CONVERSATIONS[rng.randrange(len(CONVERSATIONS))]
            origin, place = rng.choice(ORIGINS)
            prompt = ROBUST_PROMPT.format(
                setting=setting, conversation=conversation, speakers=speakers, document=document,
                persona=f"{origin} {rng.choice(['woman', 'man'])} aged {rng.randint(18, 92)}, living in {place}",
                details="; ".join(rng.sample(DETAILS, rng.randint(4, 8))))
            answer = await complete(client, prompt, max_tokens=6000, temperature=0.8, seed=7_000_000 + i)
            m = re.search(r"^\s*CONVERSATION\s*\n(.*?)\n\s*DOCUMENT\s*\n(.*?)\n\s*ICD10\s*\n(.*)$", answer, re.S | re.M)
            codes = [{"code": c, "description": d} for c, d in _ICD_LINE.findall(m.group(3))] if m else []
            if not m or not codes or len(re.findall(r"^\[[a-z_0-9]+\]", m.group(1), re.M)) < 8:
                stats["rejected"] += 1
                return None
            record = {"encounter_id": f"ROBUST-{i:04d}", "dataset": f"robust-{document.split(' (')[0][:40]}",
                      "dialogue": _ANY_TAG.sub("", m.group(1)).strip(), "note": _ANY_TAG.sub("", m.group(2)).strip(),
                      "icd10_codes": codes, "spans": []}
            for field in ("dialogue", "note"):
                record["spans"] += [{**s, "field": field} for s in await label_text(client, record[field], stats)]
            if rng.random() < 0.5:  # half of the conversations as lower-cased ASR, like ACI
                dialogue_spans = [s for s in record["spans"] if s["field"] == "dialogue"]
                record["dialogue"], dialogue_spans = to_asr(record["dialogue"], dialogue_spans)
                record["spans"] = dialogue_spans + [s for s in record["spans"] if s["field"] == "note"]
                record["dataset"] += "-asr"
            stats["accepted"] += 1
            return record

        records = [r for r in await gather_limited([one(i) for i in range(args.n)], limit=args.concurrency) if r]
    write_jsonl(args.output, [{k: v for k, v in r.items() if k != "spans"} for r in records])
    gold = [{"encounter_id": r["encounter_id"], **s, "type": s["category"]} for r in records for s in r["spans"]]
    write_jsonl(args.output.replace(".jsonl", "_gold.jsonl"), gold)
    print(f"{len(records)} records, {len(gold)} gold values -> {args.output}\n{dict(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label")
    p.add_argument("--input", default=str(ROOT / "data" / "raw" / "patient_doctor_encounters.jsonl"))
    p.add_argument("--output", default=str(OUT / "teacher_labels_heldout.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=48)
    p = sub.add_parser("generate")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default=str(OUT / "teacher_generated.jsonl"))
    p.add_argument("--concurrency", type=int, default=128)
    p = sub.add_parser("robust")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output", default=str(ROOT / "data" / "robustness" / "robustness.jsonl"))
    p.add_argument("--concurrency", type=int, default=120)
    args = parser.parse_args()
    asyncio.run({"label": label, "generate": generate, "robust": robust}[args.command](args))


if __name__ == "__main__":
    main()
