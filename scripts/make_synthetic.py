"""Synthetic encounters full of private details, with known answers (robustness to data unlike ACI-Bench).

Each synthetic record is a held-out ACI encounter (``data/raw/patient_doctor_encounters.jsonl``) that gets
* a new patient name from a European (or British/US) profile, replacing the ACI name in note and dialogue;
* 4-6 inserted dialogue exchanges and 2-3 note passages with administrative details: date of birth, phone,
  e-mail, address, insurer and insurance/AHV numbers, employer and job, relatives, other doctors and
  hospitals, appointment dates and times.

The wording of the inserted passages is written by a local LLM (Qwen3.5-9B via MLX) with placeholders
("{PHONE}"), not by us, so the test does not just replay the phrasings our rules were written for. Faker
and lists of real places fill the placeholders, so every value and its offsets are known. Half of the
dialogues keep the lower-cased ASR style of ACI, half are cased like a human transcript; in ASR style some
phone numbers, e-mails and dates are spoken ("zero seven nine ...", "anna dot keller at gmail dot com").

Two splits use disjoint LLM passages and patient profiles: ``dev`` (to look at misses) and ``test``
(reported only).

    uv run --group llm --group synth python scripts/make_synthetic.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from clinanon import resources
from clinanon.evaluation.gold import load_gold
from clinanon.io import read_jsonl, write_jsonl
from clinanon.pipeline import normalize_asr_splits

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "cache" / "synthetic"
MODEL = "mlx-community/Qwen3.5-9B-4bit"
PROMPT_VERSION = "v1"

# ------------------------------------------------------------------------------------- LLM passages

# name -> (field, what happens, placeholders that must all appear, optional ones)
SCENARIOS = {
    "identity": ("dialogue", "the doctor or nurse confirms who the patient is: their full name and date of birth",
                 ["FULL_NAME", "DOB"], []),
    "contact": ("dialogue", "the receptionist or doctor checks the patient's phone number and e-mail address",
                ["PHONE", "EMAIL"], []),
    "address": ("dialogue", "the patient's home address comes up, for example to send a letter or a referral",
                ["STREET_ADDRESS", "POSTCODE_CITY"], []),
    "moved": ("dialogue", "the patient explains they moved here recently and where they lived before",
              ["CITY"], ["COUNTRY"]),
    "insurance": ("dialogue", "the doctor or nurse checks the patient's health insurance company and numbers",
                  ["INSURER", "INSURANCE_NUMBER"], ["AHV"]),
    "work": ("dialogue", "the patient talks about their job and the company they work for, and how work affects "
             "their symptoms", ["PROFESSION", "EMPLOYER"], []),
    "family": ("dialogue", "the patient mentions a relative by first name and says how old they are, for example "
               "who drove them here or who helps at home", ["RELATIVE_NAME", "RELATIVE_AGE"], []),
    "previous_care": ("dialogue", "the patient says they were seen before at another hospital by another doctor, "
                      "and when", ["HOSPITAL", "DOCTOR_NAME", "DATE"], []),
    "follow_up": ("dialogue", "the doctor and patient agree on the date and time of the next appointment",
                  ["DATE", "TIME"], ["DOCTOR_NAME"]),
    "note_social": ("note", "a SOCIAL HISTORY paragraph: where the patient lives and with whom, their job and "
                    "employer", ["CITY", "PROFESSION", "EMPLOYER"], ["RELATIVE_NAME"]),
    "note_plan": ("note", "one or two sentences at the end of the plan: a follow-up appointment and a referral "
                  "to a named doctor at a named hospital", ["DATE", "DOCTOR_NAME", "HOSPITAL"], ["PHONE"]),
}
# Note headers are forms, not prose (and the 9B model wrote nonsense for them), so they are templates.
HEADERS = {
    "dev": [
        "Patient: {FULL_NAME}\nDOB: {DOB}\nMRN: {MRN}",
        "Name: {FULL_NAME}\nDate of birth: {DOB}\nAddress: {STREET_ADDRESS}, {POSTCODE_CITY}\nPhone: {PHONE}",
        "PATIENT: {FULL_NAME} (DOB {DOB})\nPatient ID: {MRN}\nInsurance: {INSURER}, policy no. {INSURANCE_NUMBER}",
    ],
    "test": [
        "Patient name: {FULL_NAME}\nBorn: {DOB}\nRecord no.: {MRN}\nTel.: {PHONE}",
        "{FULL_NAME}, born {DOB}\n{STREET_ADDRESS}\n{POSTCODE_CITY}\nAHV: {AHV}",
        "Name: {FULL_NAME}\nDOB: {DOB}\nHealth insurer: {INSURER} ({INSURANCE_NUMBER})\nE-mail: {EMAIL}",
    ],
}
PLACEHOLDERS = {
    "FULL_NAME": "NAME", "RELATIVE_NAME": "NAME", "DOCTOR_NAME": "NAME", "PATIENT_FIRST": "NAME",
    "PATIENT_LAST": "NAME", "DOB": "DATE", "DATE": "DATE", "TIME": "TIME", "RELATIVE_AGE": "AGE",
    "PHONE": "CONTACT", "EMAIL": "CONTACT", "STREET_ADDRESS": "LOCATION", "POSTCODE_CITY": "LOCATION",
    "CITY": "LOCATION", "COUNTRY": "LOCATION", "INSURER": "ORGANIZATION", "EMPLOYER": "ORGANIZATION",
    "HOSPITAL": "ORGANIZATION", "INSURANCE_NUMBER": "IDENTIFIER", "AHV": "IDENTIFIER", "MRN": "IDENTIFIER",
    "PROFESSION": "PROFESSION",
}  # fmt: skip
_PLACEHOLDER = re.compile(r"\{([A-Z_]+)\}")

PROMPT = """Write {n} different short excerpts from {kind}.
In each excerpt, {what}.

Rules:
{format_rule}
- Write these placeholders exactly as shown instead of the real information: {required}{optional}.
- Each placeholder stands for the complete value (for example {{DOB}} is a full date of birth, {{DOCTOR_NAME}} \
is a doctor's surname to write after "Dr."), so do not add a title, a day or a year next to it.
- Do not invent any other names, dates, numbers, places, companies or ages.
- Make the excerpts different from each other in wording and length; make them sound natural.
Separate the excerpts with a line that contains only ===. Write nothing else."""
_DIALOGUE_RULE = ("- 2 to 6 turns, each on its own line starting with [doctor] or [patient] "
                  "(the speaker in square brackets, in lower case).")
_NOTE_RULE = "- Plain text as a physician writes it in the note, no markdown, no bullet points."


def _generate(prompt: str, seed: int) -> str:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = load(MODEL)
    model, tokenizer = _MODEL
    mx.random.seed(seed)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    return generate(model, tokenizer, prompt=text, max_tokens=1800, sampler=make_sampler(temp=0.9, top_p=0.95))


def _valid(passage: str, field: str, required: list[str], optional: list[str]) -> bool:
    used = set(_PLACEHOLDER.findall(passage))
    if not set(required) <= used or not used <= set(required) | set(optional) or re.search(r"[{}]", _PLACEHOLDER.sub("", passage)):
        return False
    if field == "dialogue":
        lines = passage.splitlines()
        return 2 <= len(lines) <= 8 and all(re.match(r"^\[(?:doctor|patient)\] \S", line) for line in lines)
    return "[" not in passage and "*" not in passage and "#" not in passage


def passages(split: str, per_scenario: int, exclude: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """LLM passages per scenario (cached on disk; the dev and test splits use different seeds)."""
    out: dict[str, list[str]] = {}
    for name, (field, what, required, optional) in SCENARIOS.items():
        prompt = PROMPT.format(
            n=6, kind="a conversation between a doctor and a patient" if field == "dialogue" else "clinical notes",
            what=what, format_rule=_DIALOGUE_RULE if field == "dialogue" else _NOTE_RULE,
            required=", ".join(f"{{{p}}}" for p in required),
            optional=(", and if it fits " + ", ".join(f"{{{p}}}" for p in optional)) if optional else "",
        )
        found: list[str] = []
        for attempt in range(8):
            seed = int(hashlib.sha1(f"{split}|{name}|{attempt}".encode()).hexdigest()[:8], 16)
            path = CACHE / f"{PROMPT_VERSION}_{split}_{name}_{attempt}.json"
            if path.exists():
                answer = json.loads(path.read_text())["answer"]
            else:
                answer = _generate(prompt, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"prompt": prompt, "seed": seed, "answer": answer}, ensure_ascii=False))
            for chunk in re.split(r"^\s*===+\s*$", answer, flags=re.M):
                chunk = "\n".join(line.strip() for line in chunk.strip().splitlines() if line.strip())
                if (chunk and _valid(chunk, field, required, optional) and chunk not in found
                        and chunk not in (exclude or {}).get(name, [])):
                    found.append(chunk)
            if len(found) >= per_scenario:
                break
        print(f"  {split:4s} {name:14s} {len(found)} passages")
        out[name] = found[:per_scenario]
    return out


# ------------------------------------------------------------------------------------------- values

# Real places (a gazetteer-free detector has to recognise them from context, like on the hidden set).
CITIES = {
    "CH": ["Zürich", "Geneva", "Basel", "Lausanne", "Bern", "Winterthur", "Lucerne", "St. Gallen", "Lugano",
           "Biel", "Thun", "Bellinzona", "Fribourg", "Schaffhausen", "Chur", "Locarno", "Neuchâtel", "Sion",
           "Zug", "Aarau", "Baden", "Mendrisio", "Uster", "Wil"],
    "DE": ["Munich", "Stuttgart", "Freiburg", "Hamburg", "Cologne", "Frankfurt", "Leipzig", "Dresden"],
    "IT": ["Milan", "Como", "Varese", "Turin", "Bergamo", "Bologna", "Verona", "Genoa"],
    "FR": ["Lyon", "Annecy", "Grenoble", "Strasbourg", "Marseille", "Toulouse", "Mulhouse"],
    "GB": ["Manchester", "Leeds", "Bristol", "Edinburgh", "Glasgow", "Cardiff", "Brighton"],
    "US": ["Boston", "Chicago", "Denver", "Seattle", "Phoenix", "Atlanta", "Portland", "Austin"],
}
COUNTRIES = {"CH": "Switzerland", "DE": "Germany", "IT": "Italy", "FR": "France", "GB": "England",
             "US": "the United States"}
HOSPITALS = {
    "CH": ["Universitätsspital Zürich", "Inselspital Bern", "Kantonsspital Aarau", "Kantonsspital St. Gallen",
           "Ospedale Regionale di Lugano", "Ospedale San Giovanni", "Hôpitaux Universitaires de Genève",
           "CHUV", "Klinik Hirslanden", "Stadtspital Triemli", "Universitätsspital Basel",
           "Luzerner Kantonsspital", "Spital Bülach", "Kantonsspital Winterthur", "Clinica Luganese Moncucco"],
    "DE": ["Charité", "Klinikum rechts der Isar", "Universitätsklinikum Heidelberg", "Robert-Bosch-Krankenhaus"],
    "IT": ["Ospedale Niguarda", "Policlinico di Milano", "Ospedale Sant'Anna"],
    "FR": ["Hôpital Édouard Herriot", "CHU Grenoble Alpes", "Hôpital de la Croix-Rousse"],
    "GB": ["St Thomas' Hospital", "Royal Infirmary of Edinburgh", "Leeds General Infirmary"],
    "US": ["Mayo Clinic", "Johns Hopkins Hospital", "Cleveland Clinic", "Massachusetts General Hospital",
           "Mount Sinai Hospital"],
}
INSURERS = {
    "CH": ["Helsana", "CSS", "Swica", "Sanitas", "Concordia", "Visana", "Groupe Mutuel", "Assura", "KPT",
           "Atupri", "Sympany", "ÖKK"],
    "DE": ["AOK", "Techniker Krankenkasse", "Barmer", "DAK-Gesundheit"],
    "IT": ["Generali", "UniSalute"], "FR": ["MGEN", "Harmonie Mutuelle"], "GB": ["Bupa", "AXA Health"],
    "US": ["Blue Cross Blue Shield", "Aetna", "Cigna", "UnitedHealthcare", "Kaiser Permanente", "Humana"],
}
EMPLOYERS = ["Migros", "Coop", "Swisscom", "Roche", "Novartis", "UBS", "SBB", "Nestlé", "ABB", "Bosch",
             "Swiss Post", "Zurich Insurance", "Lidl", "Siemens", "Manor", "Logitech", "Credit Suisse",
             "Swatch", "Raiffeisen", "Holcim"]
# Written independently of src/clinanon/resources/occupations.txt.
PROFESSIONS = [
    "bank clerk", "electrical engineer", "kindergarten teacher", "train conductor", "hotel receptionist",
    "watchmaker", "pharmacist", "IT consultant", "postman", "translator", "civil engineer", "tram driver",
    "dental assistant", "accountant", "carpenter", "graphic designer", "sommelier", "ski instructor",
    "mountain guide", "chef", "journalist", "nurse", "software developer", "logistics manager", "cheesemaker",
    "insurance broker", "lab technician", "architect", "high school teacher", "social worker",
    "police officer", "plumber", "hairdresser", "flight attendant", "physiotherapist", "bus driver",
    "sales manager", "cleaner", "warehouse supervisor", "primary school teacher",
]  # fmt: skip
PROFILES = [  # (country, Faker locale for names/streets/phones, weight)
    ("CH", "de_CH", 4), ("CH", "fr_CH", 2), ("CH", "it_IT", 2), ("DE", "de_DE", 1), ("IT", "it_IT", 1),
    ("FR", "fr_FR", 1), ("GB", "en_GB", 1), ("US", "en_US", 1),
]
EMAIL_DOMAINS = {"CH": ["bluewin.ch", "gmx.ch", "sunrise.ch", "gmail.com", "hotmail.com"],
                 "DE": ["gmx.de", "web.de", "gmail.com"], "IT": ["libero.it", "gmail.com"],
                 "FR": ["orange.fr", "free.fr", "gmail.com"], "GB": ["btinternet.com", "gmail.com"],
                 "US": ["gmail.com", "yahoo.com", "outlook.com"]}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
             "nine": "ninth", "twelve": "twelfth"}


def _spoken_number(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")


def _spoken_ordinal(n: int) -> str:
    words = _spoken_number(n)
    head, _, last = words.rpartition("-")
    last = _ORDINALS.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return f"{head}-{last}" if head else last


def _spoken_year(y: int) -> str:
    if 2000 <= y < 2010:
        return "two thousand" + (f" {_ONES[y - 2000]}" if y > 2000 else "")
    hi, lo = divmod(y, 100)
    return f"{_spoken_number(hi)} {('oh ' + _ONES[lo]) if 0 < lo < 10 else _spoken_number(lo)}"


def _ahv(rng: random.Random) -> str:
    digits = [7, 5, 6] + [rng.randrange(10) for _ in range(9)]
    check = (10 - sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits)) % 10) % 10  # EAN-13
    s = "".join(map(str, digits + [check]))
    return f"{s[:3]}.{s[3:7]}.{s[7:11]}.{s[11:]}"


class Profile:
    """One fake patient with everything the placeholders can ask for."""

    def __init__(self, rng: random.Random, gender: str, age: int, visit: date):
        country, locale, _ = rng.choices(PROFILES, weights=[w for *_, w in PROFILES])[0]
        fake = Faker(locale)
        fake.seed_instance(rng.randrange(2**31))
        self.rng, self.fake, self.country = rng, fake, country
        self.first = fake.first_name_female() if gender == "female" else fake.first_name_male()
        self.last = fake.last_name()
        self.dob = visit - timedelta(days=365 * age + rng.randrange(365))
        self.visit = visit
        self.city = rng.choice(CITIES[country])
        self.relative_names = [fake.first_name() for _ in range(3)]
        self.doctor_names = [fake.last_name() for _ in range(3)]

    def value(self, placeholder: str, spoken: bool) -> str:
        rng, fake = self.rng, self.fake
        if placeholder == "FULL_NAME":
            return f"{self.first} {self.last}"
        if placeholder == "RELATIVE_NAME":
            return rng.choice(self.relative_names)
        if placeholder == "DOCTOR_NAME":
            return rng.choice(self.doctor_names)
        if placeholder == "DOB":
            return self._date(self.dob, spoken, with_year=True)
        if placeholder == "DATE":
            when = self.visit + timedelta(days=rng.choice([-1, 1]) * rng.randrange(7, 400))
            return self._date(when, spoken, with_year=rng.random() < 0.4)
        if placeholder == "TIME":
            h, m = rng.choice(range(7, 18)), rng.choice([0, 15, 30, 45])
            hour = _spoken_number(h % 12 or 12)
            if spoken:
                return {0: f"{hour} o'clock", 15: f"quarter past {hour}", 30: f"half past {hour}",
                        45: f"{hour} forty-five"}[m]
            return rng.choice([f"{h}:{m:02d}", f"{h % 12 or 12}:{m:02d} {'am' if h < 12 else 'pm'}",
                               f"{h % 12 or 12} {'am' if h < 12 else 'pm'}" if m == 0 else f"{h}.{m:02d}"])
        if placeholder == "RELATIVE_AGE":
            return str(rng.randrange(8, 90)) if not spoken else _spoken_number(rng.randrange(8, 90))
        if placeholder == "PHONE":
            number = fake.phone_number()
            if spoken:
                digits = re.sub(r"\D", "", number.replace("+41", "0").replace("+49", "0").replace("+33", "0"))
                return " ".join(_DIGITS[int(d)] for d in digits[:10])
            return number
        if placeholder == "EMAIL":
            user = rng.choice([f"{self.first}.{self.last}", f"{self.first[0]}{self.last}",
                               f"{self.first}{rng.randrange(10, 99)}"])
            user = re.sub(r"[^a-z0-9.]", "", user.lower().translate(str.maketrans("äöüéèàç", "aoueeac")))
            domain = rng.choice(EMAIL_DOMAINS[self.country])
            if spoken:
                return f"{user.replace('.', ' dot ')} at {domain.replace('.', ' dot ')}"
            return f"{user}@{domain}"
        if placeholder == "STREET_ADDRESS":
            street = fake.street_address()
            if self.country == "CH" and not re.search(r"\d", street):
                street += f" {rng.randrange(1, 120)}"
            return street
        if placeholder == "POSTCODE_CITY":
            return f"{fake.postcode()} {self.city}" if self.country != "US" else f"{self.city}, {fake.state_abbr()} {fake.postcode()}"
        if placeholder == "CITY":
            return self.city if rng.random() < 0.5 else rng.choice(CITIES[rng.choice(list(CITIES))])
        if placeholder == "COUNTRY":
            return COUNTRIES[rng.choice([c for c in COUNTRIES if c != self.country])]
        if placeholder == "INSURER":
            return rng.choice(INSURERS[self.country])
        if placeholder == "HOSPITAL":
            return rng.choice(HOSPITALS[self.country] if rng.random() < 0.7 else HOSPITALS[rng.choice(list(HOSPITALS))])
        if placeholder == "EMPLOYER":
            return rng.choice(EMPLOYERS) if rng.random() < 0.6 else fake.company()
        if placeholder == "PROFESSION":
            return rng.choice(PROFESSIONS)
        if placeholder == "INSURANCE_NUMBER":
            return rng.choice([fake.bothify("80756########"), fake.bothify("###.####.###"),
                               fake.bothify("??#######").upper(), fake.bothify("#### #### ####")])
        if placeholder == "AHV":
            return _ahv(rng)
        if placeholder == "MRN":
            return rng.choice([fake.bothify("#######"), fake.bothify("MRN-######"), fake.bothify("PID ########")])
        raise KeyError(placeholder)

    def _date(self, d: date, spoken: bool, with_year: bool) -> str:
        rng, month = self.rng, _MONTHS[d.month - 1]
        if spoken:
            day = rng.choice([_spoken_ordinal(d.day), f"the {_spoken_ordinal(d.day)} of"])
            text = f"{month} {day}" if not day.startswith("the") else f"{day} {month}"
            return f"{text} {_spoken_year(d.year)}" if with_year else text
        if with_year:
            return rng.choice([f"{d.day} {month} {d.year}", f"{d.day:02d}.{d.month:02d}.{d.year}",
                               f"{d.month:02d}/{d.day:02d}/{d.year}", f"{month} {d.day}, {d.year}",
                               f"{d.day}.{d.month}.{d.year}", f"{d.year}-{d.month:02d}-{d.day:02d}"])
        return rng.choice([f"{d.day} {month}", f"{month} {d.day}", f"the {d.day}th of {month}".replace("1th", "1st")
                           .replace("2th", "2nd").replace("3th", "3rd").replace("11st", "11th")
                           .replace("12nd", "12th").replace("13rd", "13th")])


# ------------------------------------------------------------------------------------------ records

Segment = tuple[str, str | None]  # (text, placeholder type of a gold value or None)


def _asr(text: str) -> str:
    """ACI ASR style: lower case, punctuation separated by spaces."""
    text = re.sub(r"\s*([,.?!])(?=\s|$)", r" \1", text.lower())
    return re.sub(r"[;:\"“”]", "", text)


def _fill(passage: str, profile: Profile, asr: bool) -> list[Segment]:
    segments: list[Segment] = []
    pos = 0
    for m in _PLACEHOLDER.finditer(passage):
        before = passage[pos : m.start()]
        segments.append((_asr(before) if asr else before, None))
        spoken = asr and m.group(1) in {"PHONE", "EMAIL", "DOB", "DATE", "TIME", "RELATIVE_AGE"} and profile.rng.random() < 0.5
        value = profile.value(m.group(1), spoken)
        if value.lower().startswith("the "):  # "the United States", "the 5th of April": "the" is not private
            segments.append((value[:4].lower() if asr else value[:4], None))
            value = value[4:]
        segments.append((value.lower() if asr else value, m.group(1)))
        pos = m.end()
    tail = passage[pos:]
    segments.append((_asr(tail) if asr else tail, None))
    return segments


def _swap_names(text: str, old: list[str], new: list[str], style: str) -> list[Segment]:
    """Replace the ACI patient name (first/last) by the profile's.

    ``style``: "note" (only capitalized occurrences, casing kept), "asr" (lower case), "cased" (capitalized).
    """
    pattern = re.compile(r"\b(" + "|".join(re.escape(o) for o in old) + r")\b", re.I)
    segments: list[Segment] = []
    pos = 0
    for m in pattern.finditer(text):
        if style == "note" and not m.group()[0].isupper():
            continue  # "Brown" the patient vs "brown" the colour
        replacement = new[[o.lower() for o in old].index(m.group().lower())]
        if style == "asr" or (style == "note" and not m.group()[0].isupper()):
            replacement = replacement.lower()
        segments.append((text[pos : m.start()], None))
        segments.append((replacement, "PATIENT_FIRST" if replacement.lower() == new[0].lower() else "PATIENT_LAST"))
        pos = m.end()
    segments.append((text[pos:], None))
    return segments


def _case_like_transcript(text: str) -> str:
    """Same-length cased version of a lower-cased ASR dialogue (sentence starts, "I"); values keep their case."""
    out = re.sub(r"(^\[[a-z]+\]\s+|[.?!]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text, flags=re.M)
    return re.sub(r"\bi\b", "I", out)


def _join(segments: list[Segment]) -> tuple[str, list[dict]]:
    text, gold = "", []
    for piece, kind in segments:
        if kind and piece.strip():
            gold.append({"start": len(text), "end": len(text) + len(piece), "text": piece, "type": kind,
                         "category": PLACEHOLDERS.get(kind, "NAME")})
        text += piece
    return text, gold


def build_record(base: dict, gold_ids, rng: random.Random, pools: dict[str, list[str]], asr: bool,
                 eid: str) -> tuple[dict, list[dict]]:
    g = gold_ids
    age = int(g.age) if g.age and g.age.isdigit() else rng.randrange(25, 80)
    profile = Profile(rng, "female" if rng.random() < 0.5 else "male", age, date(2026, 3, 2) + timedelta(rng.randrange(200)))
    old = [g.first_name, g.last_name]
    new = [profile.first, profile.last]

    note = normalize_asr_splits(base["note"])
    dialogue = normalize_asr_splits(base["dialogue"])
    if not asr:
        dialogue = _case_like_transcript(dialogue)

    # Dialogue: name swap per turn, then insert exchanges between turns (identity early, follow-up late).
    turns = dialogue.split("\n")
    turn_segments = [_swap_names(t, old, new, style="asr" if asr else "cased") for t in turns]
    names = ["identity"] + rng.sample(["contact", "address", "moved", "insurance", "work", "family",
                                       "previous_care"], rng.randrange(3, 5)) + ["follow_up"]
    slots = sorted(rng.sample(range(2, max(len(turns) - 1, 4)), len(names) - 2))
    positions = [1, *slots, len(turns) - 1]
    for name, at in sorted(zip(names, positions, strict=True), key=lambda x: -x[1]):
        segments = _fill(rng.choice(pools[name]), profile, asr)
        if not asr:
            segments = [(_case_like_transcript(t) if k is None else t, k) for t, k in segments]
        turn_segments.insert(at, segments)
    dialogue_segments: list[Segment] = []
    for i, segs in enumerate(turn_segments):
        dialogue_segments.extend(segs if i == 0 else [("\n", None), *segs])

    # Note: header on top, social history after the history of present illness (or the first section),
    # plan sentence at the end.
    header = _fill(rng.choice(pools["note_header"]), profile, asr=False)
    social = _fill(rng.choice(pools["note_social"]), profile, asr=False)
    plan = _fill(rng.choice(pools["note_plan"]), profile, asr=False)
    hpi = max(note.find("HISTORY OF PRESENT ILLNESS"), 0)
    m = re.compile(r"\n\n(?=[A-Z][A-Z /&]+\n)").search(note, hpi + 30)
    cut = m.start() if m else len(note)
    note_segments = [*header, ("\n\n", None), *_swap_names(note[:cut], old, new, style="note"),
                     ("\n\nSOCIAL HISTORY\n\n", None), *social,
                     *_swap_names(note[cut:], old, new, style="note"), ("\n\n", None), *plan]

    dialogue_text, dialogue_gold = _join(dialogue_segments)
    note_text, note_gold = _join(note_segments)
    assert normalize_asr_splits(dialogue_text) == dialogue_text and normalize_asr_splits(note_text) == note_text
    record = {"encounter_id": eid, "dataset": f"synthetic-{'asr' if asr else 'cased'}", "dialogue": dialogue_text,
              "note": note_text, "icd10_codes": base["icd10_codes"]}
    gold = [{**item, "field": "dialogue"} for item in dialogue_gold] + [{**item, "field": "note"} for item in note_gold]
    for item in gold:
        assert record[item["field"]][item["start"] : item["end"]] == item["text"]
    return record, gold



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=str(ROOT / "data" / "raw" / "patient_doctor_encounters.jsonl"))
    parser.add_argument("--n", type=int, default=40, help="records per split")
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    args = parser.parse_args()

    gold_ids = load_gold()
    common = resources.common_words()
    # Patients whose names are ordinary words ("Will", "Brown") would make the name swap ambiguous.
    bases = [r for r in read_jsonl(args.base)
             if gold_ids[r["encounter_id"]].first_name and gold_ids[r["encounter_id"]].last_name
             and not {gold_ids[r["encounter_id"]].first_name.lower(), gold_ids[r["encounter_id"]].last_name.lower()} & common]
    print(f"{len(bases)} usable base encounters")
    dev_pools: dict[str, list[str]] = {}
    for split, seed in (("dev", 1), ("test", 2)):
        pools = passages(split, per_scenario=10, exclude=dev_pools)
        dev_pools = dev_pools or pools
        pools["note_header"] = HEADERS[split]
        rng = random.Random(seed)
        records, gold = [], []
        for i, base in enumerate(rng.sample(bases, args.n)):
            eid = f"SYN-{split}-{i:03d}"
            record, items = build_record(base, gold_ids[base["encounter_id"]], rng, pools, asr=i % 2 == 0, eid=eid)
            records.append(record)
            gold.extend({"encounter_id": eid, "base": base["encounter_id"], **item} for item in items)
        out = Path(args.out)
        write_jsonl(out / f"synthetic_{split}.jsonl", records)
        write_jsonl(out / f"synthetic_{split}_gold.jsonl", gold)
        print(f"{split}: {len(records)} records, {len(gold)} gold values -> {out}")


if __name__ == "__main__":
    main()
