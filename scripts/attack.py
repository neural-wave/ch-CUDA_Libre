"""Red-team our own output with the teacher LLM (bonus task: adversarial privacy attacks).

recover  Every tag in the anonymized encounter gets a number ([AGE#3]); the attacker sees the whole
         anonymized encounter and guesses the original value behind each one. We score the guesses
         against the true originals (from the decision log): exact match, and approximate match (an age
         within 3 years, a date with the same year or month and day).
link     Re-identification: the attacker sees one anonymized encounter and 10 candidate patients (name,
         age, sex; the true one and 9 others from the ACI metadata) and must pick the right one. Chance is
         10%; sex is not removed on purpose (clinically relevant), so the floor is about 20%.
profile  Linkage with background knowledge: from each ORIGINAL encounter the teacher writes what an
         acquaintance or a social-media post could know about the person, non-medical facts only (age, job,
         family, hobbies, places, trips; no names, no diagnoses). The attacker then rates every profile
         against every anonymized encounter (n x n) and we count how often the true encounter ranks first
         (chance 1/n). Run on the original texts too, to see what anonymization prevents.

    TEACHER_URL=http://localhost:8000/v1 uv run python scripts/attack.py recover \\
        --original data/raw/eval_patient_doctor_encounters.jsonl --decisions submissions/final/eval_decisions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from clinanon.distill.teacher import complete, gather_limited
from clinanon.evaluation.gold import load_gold
from clinanon.io import read_jsonl
from clinanon.pipeline import normalize_asr_splits

ROOT = Path(__file__).resolve().parents[1]

RECOVER_PROMPT = """You are a privacy red-teamer. The clinical text below was anonymized: private details were \
replaced by numbered placeholders such as [AGE#3], [PATIENT#1] or [LOCATION#7].

Use every clue in the text (medical history, life events, family, work, seasons, word choice, anything) to \
guess the original value behind each placeholder. Guess even when unsure.

Answer with JSON only: an object mapping each placeholder number to your guess, for example \
{"1": "Martha", "3": "58", "7": "Seattle"}.

DIALOGUE:
%s

NOTE:
%s"""

LINK_PROMPT = """You are a privacy red-teamer. Below is an anonymized clinical encounter, followed by 10 candidate \
patients. Exactly one of them is the patient of this encounter. Use every clue in the text (age clues, sex, \
life situation, anything) to pick the most likely candidate.

Answer with JSON only: {"candidate": <number>, "reason": "<one sentence>"}.

DIALOGUE:
%s

NOTE:
%s

CANDIDATES:
%s"""


def build_attack_view(original: dict, decisions: list[dict], counter: list[int]) -> tuple[dict, dict[int, dict]]:
    """Anonymized copy of the encounter with numbered tags, built from the original and its decisions."""
    view, answers = {}, {}
    for field in ("dialogue", "note"):
        text = normalize_asr_splits(original[field])  # decision offsets refer to the normalized text
        tags = sorted((d for d in decisions if d["field"] == field and d["action"] == "tag"),
                      key=lambda d: d["start"])
        out, pos = [], 0
        for d in tags:
            if d["start"] < pos:
                continue
            counter[0] += 1
            n = counter[0]
            out.append(text[pos : d["start"]])
            out.append(f"{d['replacement'][:-1]}#{n}]")
            answers[n] = {"category": d["category"], "original": d["text"], "replacement": d["replacement"]}
            pos = d["end"]
        out.append(text[pos:])
        view[field] = "".join(out)
    return view, answers


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


_NUMBER_WORDS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _number(value: str) -> int | None:
    m = re.search(r"\d{1,3}", value)
    if m:
        return int(m.group())
    words = _norm(value).replace("-", " ").split()
    total = None
    for w in words:
        if w in _TENS:
            total = (total or 0) + _TENS[w]
        elif w in _NUMBER_WORDS:
            total = (total or 0) + _NUMBER_WORDS[w]
    return total


def judge(category: str, original: str, guess: str) -> tuple[bool, bool]:
    """(exact, approximate) recovery of one value."""
    o, g = _norm(original), _norm(guess)
    if not g:
        return False, False
    exact = o == g or (len(g) > 2 and (o in g.split() or g in o.split() or o == g))
    approx = exact
    if category == "AGE":
        a, b = _number(original), _number(guess)
        approx = a is not None and b is not None and abs(a - b) <= 3
    elif category == "DATE":
        years_o, years_g = set(re.findall(r"(?:19|20)\d\d", original)), set(re.findall(r"(?:19|20)\d\d", guess))
        approx = exact or bool(years_o & years_g) or (bool(set(o.split()) & set(g.split()) - {"the", "of"}))
    elif category in ("NAME", "LOCATION", "ORGANIZATION", "PROFESSION"):
        approx = exact or bool(set(o.split()) & set(g.split()) - {"the", "of", "dr", "mr", "mrs", "ms"})
    return exact, approx


def _json(answer: str) -> dict:
    m = re.search(r"\{.*\}", answer, flags=re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


async def recover(args) -> None:
    originals = {r["encounter_id"]: r for r in read_jsonl(args.original)}
    decisions = defaultdict(list)
    for d in read_jsonl(args.decisions):
        decisions[d["encounter_id"]].append(d)
    rows = []
    async with httpx.AsyncClient() as client:

        async def one(eid: str):
            view, answers = build_attack_view(originals[eid], decisions[eid], [0])
            if not answers:
                return []
            guess = _json(await complete(client, RECOVER_PROMPT % (view["dialogue"], view["note"]),
                                         max_tokens=4000, thinking=args.thinking))
            out = []
            for n, a in answers.items():
                g = str(guess.get(str(n), ""))
                exact, approx = judge(a["category"], a["original"], g)
                out.append({"encounter_id": eid, "n": n, **a, "guess": g, "exact": exact, "approx": approx})
            return out

        for result in await gather_limited([one(eid) for eid in originals], limit=args.concurrency):
            rows.extend(result)
    counts = defaultdict(Counter)
    for r in rows:
        key = r["category"]
        counts[key]["n"] += 1
        counts[key]["exact"] += r["exact"]
        counts[key]["approx"] += r["approx"]
    total = Counter()
    print(f"{'category':14s} {'tags':>6s} {'exact %':>8s} {'approx %':>9s}")
    for key, c in sorted(counts.items(), key=lambda kv: -kv[1]["n"]):
        total.update(c)
        print(f"{key:14s} {c['n']:6d} {100 * c['exact'] / c['n']:8.1f} {100 * c['approx'] / c['n']:9.1f}")
    print(f"{'ALL':14s} {total['n']:6d} {100 * total['exact'] / total['n']:8.1f} {100 * total['approx'] / total['n']:9.1f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"details -> {out}")


async def link(args) -> None:
    originals = {r["encounter_id"]: r for r in read_jsonl(args.original)}
    decisions = defaultdict(list)
    for d in read_jsonl(args.decisions):
        decisions[d["encounter_id"]].append(d)
    gold = load_gold()
    pool = [(eid, g) for eid, g in gold.items() if g.first_name and g.age]
    sexes = {}
    for path in sorted((ROOT / "data" / "external" / "aci_bench_metadata").glob("*.csv")):
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sexes[row["encounter_id"]] = row["patient_gender"]
    rng = random.Random(7)
    hits = Counter()
    async with httpx.AsyncClient() as client:

        async def one(eid: str):
            if eid not in gold or not gold[eid].first_name or not gold[eid].age:
                return None
            view, _ = build_attack_view(originals[eid], decisions[eid], [0])
            view = {f: re.sub(r"#\d+\]", "]", t) for f, t in view.items()}
            others = rng.sample([p for p in pool if p[0] != eid], 9)
            candidates = others + [(eid, gold[eid])]
            rng.shuffle(candidates)
            listing = "\n".join(f"{i + 1}. {g.first_name} {g.last_name}, {g.age} years, {sexes.get(e, '?')}"
                                for i, (e, g) in enumerate(candidates))
            answer = _json(await complete(client, LINK_PROMPT % (view["dialogue"], view["note"], listing),
                                          max_tokens=2000, thinking=args.thinking))
            try:
                pick = candidates[int(answer.get("candidate")) - 1][0]
            except (TypeError, ValueError, IndexError):
                return "invalid"
            return pick == eid

        results = await gather_limited([one(eid) for eid in originals], limit=args.concurrency)
    for r in results:
        if r is not None:
            hits[r] += 1
    n = hits[True] + hits[False] + hits["invalid"]
    print(f"linkage attack: {hits[True]}/{n} encounters re-identified ({100 * hits[True] / n:.1f}%; chance 10%), "
          f"{hits['invalid']} invalid answers")


PROFILE_PROMPT = """Below is a clinical encounter. Write two or three sentences that an acquaintance, a neighbour \
or the person's own social-media posts could reveal about this patient, using only NON-medical facts that the \
encounter mentions: approximate age, sex, job, family, hobbies, pets, places, trips, events, dates.

Do not mention any symptom, diagnosis, body part, test, medication or treatment, and do not mention any name. \
If the encounter reveals little, write little. Answer with the sentences only.

DIALOGUE:
%s

NOTE:
%s"""
MATCH_PROMPT = """DIALOGUE:
%s

NOTE:
%s

The clinical encounter above was anonymized. Here is what is publicly known about a person:
"%s"

How likely is it that this person is the patient of the encounter? Use every clue (age, sex, job, family, \
hobbies, places, events). Answer with one number from 0 (impossible) to 100 (certain), nothing else."""


async def profile_attack(args) -> None:
    originals = read_jsonl(args.original)
    ids = [r["encounter_id"] for r in originals]
    limits = httpx.Limits(max_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        profiles = await gather_limited(
            [complete(client, PROFILE_PROMPT % (r["dialogue"], r["note"]), max_tokens=300) for r in originals],
            limit=args.concurrency)
        out = Path(args.out).with_name("profiles.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"encounter_id": i, "profile": p.strip()}) + "\n"
                               for i, p in zip(ids, profiles, strict=True)))
        variants = {"original": originals}
        for spec in args.variants:
            name, path = spec.split("=", 1)
            by_id = {r["encounter_id"]: r for r in read_jsonl(path)}
            variants[name] = [by_id[i] for i in ids]
        for name, records in variants.items():

            async def score(j: int, i: int, records=records):
                r = records[j]
                answer = await complete(client, MATCH_PROMPT % (r["dialogue"], r["note"], profiles[i].strip()),
                                        max_tokens=8)
                m = re.search(r"\d{1,3}", answer)
                return int(m.group()) if m else 0

            n = len(records)
            # encounter-major order, so consecutive requests share the encounter as a cached prefix
            flat = await gather_limited([score(j, i) for j in range(n) for i in range(n)], limit=args.concurrency)
            matrix = [[flat[j * n + i] for j in range(n)] for i in range(n)]  # matrix[profile][encounter]
            top1 = ranks = 0
            for i, row in enumerate(matrix):
                higher = sum(s > row[i] for s in row)
                ties = sum(s == row[i] for s in row) - 1
                rank = 1 + higher + ties / 2  # expected rank when ties are broken at random
                ranks += 1 / rank
                top1 += (higher == 0) / (ties + 1)
            print(f"{name:10s} profile linkage: top-1 {100 * top1 / n:5.1f}% (chance {100 / n:.1f}%), "
                  f"mean reciprocal rank {ranks / n:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["recover", "link", "profile"])
    parser.add_argument("--original", default=str(ROOT / "data" / "raw" / "eval_patient_doctor_encounters.jsonl"))
    parser.add_argument("--decisions", default=str(ROOT / "submissions" / "final" / "eval_decisions.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "attack" / "recover.jsonl"))
    parser.add_argument("--thinking", action="store_true", help="let the attacker reason before answering")
    parser.add_argument("--variants", nargs="*", default=[], help="profile attack: name=anonymized.jsonl pairs")
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run({"recover": recover, "link": link, "profile": profile_attack}[args.command](args))


if __name__ == "__main__":
    main()
