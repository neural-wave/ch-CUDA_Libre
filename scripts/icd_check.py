"""Estimate the utility metric: ICD-10 coding F1 on original vs anonymized text.

The teacher LLM acts as a clinical coder: it reads an encounter (dialogue and note, without the codes) and
lists ICD-10-CM codes. We score its codes against the gold codes, once on the original text and once on
each anonymized version; the F1 delta is the cost of anonymization for ICD coding. Codes are compared
exactly and at category level (first three characters, "M17.0" -> "M17").

    TEACHER_URL=http://localhost:8000/v1 uv run python scripts/icd_check.py \\
        --original data/raw/eval_patient_doctor_encounters.jsonl \\
        --anonymized midterm=submissions/midterm/eval_patient_doctor_encounters_anonymized.jsonl \\
                     final=submissions/final/eval_patient_doctor_encounters_anonymized.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import httpx

from clinanon.distill.teacher import complete, gather_limited
from clinanon.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]

PROMPT = """You are an experienced clinical coder. Read this clinical encounter (a doctor-patient conversation \
and the doctor's note) and assign the ICD-10-CM codes for the conditions it documents, including external \
causes, activities and status codes where the text supports them. Some private details are replaced by \
placeholders such as [PATIENT] or [DATE]; ignore them.

Answer with JSON only: a list of codes, for example ["M17.0", "E11.9"].

DIALOGUE:
%s

NOTE:
%s"""


def _codes(answer: str) -> list[str]:
    m = re.search(r"\[.*?\]", answer, flags=re.S)
    try:
        items = json.loads(m.group()) if m else []
    except json.JSONDecodeError:
        items = re.findall(r"[A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?", answer)
    return sorted({str(c).strip().upper() for c in items if re.fullmatch(r"[A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?", str(c).strip().upper())})


def f1(pred: list[set[str]], gold: list[set[str]]) -> dict:
    tp = sum(len(p & g) for p, g in zip(pred, gold, strict=True))
    fp = sum(len(p - g) for p, g in zip(pred, gold, strict=True))
    fn = sum(len(g - p) for p, g in zip(pred, gold, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


async def code_all(records: list[dict], concurrency: int, runs: int) -> list[set[str]]:
    """Codes per record; with runs > 1 the union over seeds is replaced by a majority vote."""
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 8)) as client:

        async def one(r):
            votes: Counter = Counter()
            for seed in range(runs):
                answer = await complete(client, PROMPT % (r["dialogue"], r["note"]), max_tokens=800,
                                        temperature=0.0 if runs == 1 else 0.6, seed=seed)
                votes.update(_codes(answer))
            return {c for c, n in votes.items() if n * 2 > runs}

        return await gather_limited([one(r) for r in records], limit=concurrency)


async def main_async(args) -> None:
    originals = read_jsonl(args.original)
    gold = [{c["code"].upper() for c in r["icd10_codes"]} for r in originals]
    variants = {"original": originals}
    for spec in args.anonymized:
        name, path = spec.split("=", 1)
        by_id = {r["encounter_id"]: r for r in read_jsonl(path)}
        variants[name] = [by_id[r["encounter_id"]] for r in originals]
    results, base = {}, None
    print(f"{'text':14s} {'F1':>7s} {'P':>7s} {'R':>7s} {'F1 (category)':>14s} {'delta F1':>9s}")
    for name, records in variants.items():
        pred = await code_all(records, args.concurrency, args.runs)
        exact = f1(pred, gold)
        category = f1([{c[:3] for c in p} for p in pred], [{c[:3] for c in g} for g in gold])
        base = base or exact["f1"]
        results[name] = {"exact": exact, "category": category, "delta_f1": base - exact["f1"],
                         "predictions": [sorted(p) for p in pred]}
        print(f"{name:14s} {exact['f1']:7.3f} {exact['precision']:7.3f} {exact['recall']:7.3f} "
              f"{category['f1']:14.3f} {base - exact['f1']:9.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"details -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", default=str(ROOT / "data" / "raw" / "eval_patient_doctor_encounters.jsonl"))
    parser.add_argument("--anonymized", nargs="+", default=[], help="name=path pairs")
    parser.add_argument("--runs", type=int, default=1, help="majority vote over this many sampled answers")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "icd_check" / "eval.json"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
