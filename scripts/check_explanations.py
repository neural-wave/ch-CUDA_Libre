"""Check a sample of decisions and their explanations with the teacher LLM (bonus task: explainability).

Stratified sample from a decisions file: removals of every category and discarded candidates ("vetoes").
For each, the judge sees the context with the span marked, the decision and its explanation, and answers
whether the decision is right (private information removed / clinical or harmless text kept) and whether
the explanation is accurate.

    TEACHER_URL=http://localhost:8000/v1 uv run python scripts/check_explanations.py \\
        --original data/raw/eval_patient_doctor_encounters.jsonl --decisions submissions/final/eval_decisions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from clinanon.distill.teacher import complete, gather_limited
from clinanon.io import read_jsonl
from clinanon.pipeline import normalize_asr_splits

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """You are auditing an anonymization tool for clinical text. Its goal: remove information that could \
identify a person (names, ages, dates, places, organizations, contact details, ID numbers, jobs), and keep \
clinical information (diagnoses, eponyms like "Parkinson's disease", drugs, relative time like "yesterday", \
body parts, measurements).

Context (the span is marked with « »):
{context}

Decision: {decision}
Explanation given by the tool: {explanation}

Is the decision right for this span, and is the explanation accurate (does it describe correctly what the \
span is and why)? Answer with JSON only: {{"decision_correct": true or false, "explanation_correct": true or \
false, "comment": "<one short sentence>"}}"""


def sample(decisions: list[dict], n_tags: int, n_vetoes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tags = defaultdict(list)
    for d in decisions:
        if d["action"] == "tag":
            tags[d["category"]].append(d)
    picked = []
    while len(picked) < n_tags and any(tags.values()):  # round-robin over categories
        for cat in sorted(tags):
            if tags[cat] and len(picked) < n_tags:
                picked.append(tags[cat].pop(rng.randrange(len(tags[cat]))))
    vetoes = [d for d in decisions if d["action"] != "tag"]
    return picked + rng.sample(vetoes, min(n_vetoes, len(vetoes)))


async def main_async(args) -> None:
    originals = {r["encounter_id"]: r for r in read_jsonl(args.original)}
    chosen = sample(read_jsonl(args.decisions), args.tags, args.vetoes, args.seed)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=80)) as client:

        async def one(d: dict):
            text = normalize_asr_splits(originals[d["encounter_id"]][d["field"]])
            context = text[max(0, d["start"] - 250) : d["start"]] + "«" + text[d["start"] : d["end"]] + "»" + \
                text[d["end"] : d["end"] + 150]
            if d["action"] == "tag":
                decision = f"removed and replaced by {d['replacement']}"
            elif "removed anyway" in (d.get("explanation") or ""):
                decision = "this detection was rejected, but another detector removed the span"
            else:
                decision = "kept (not removed)"
            answer = await complete(client, PROMPT.format(context=context, decision=decision,
                                                          explanation=d.get("explanation") or d["reason"]),
                                    max_tokens=8000, thinking=True)
            m = re.search(r"\{.*\}", answer, flags=re.S)
            try:
                verdict = json.loads(m.group()) if m else {}
            except json.JSONDecodeError:
                verdict = {}
            return {**d, **{k: verdict.get(k) for k in ("decision_correct", "explanation_correct", "comment")}}

        rows = await gather_limited([one(d) for d in chosen], limit=64)
    stats = defaultdict(Counter)
    for r in rows:
        group = "removed" if r["action"] == "tag" else "kept"
        stats[group]["n"] += 1
        stats[group]["decision"] += r["decision_correct"] is True
        stats[group]["explanation"] += r["explanation_correct"] is True
    for group, c in stats.items():
        print(f"{group:8s} n={c['n']:3d}  decision correct {100 * c['decision'] / c['n']:5.1f}%  "
              f"explanation correct {100 * c['explanation'] / c['n']:5.1f}%")
    # Weighted by how often each (action, category) occurs in the whole file, over answered items only.
    population = Counter((d["action"] == "tag", d["category"]) for d in read_jsonl(args.decisions))
    sampled = Counter((r["action"] == "tag", r["category"]) for r in rows if r["decision_correct"] is not None)
    for is_tag in (True, False):
        num = den = num_e = 0.0
        for r in rows:
            key = (r["action"] == "tag", r["category"])
            if key[0] != is_tag or r["decision_correct"] is None:
                continue
            w = population[key] / sampled[key]
            den += w
            num += w * (r["decision_correct"] is True)
            num_e += w * (r["explanation_correct"] is True)
        if den:
            print(f"{'removed' if is_tag else 'kept':8s} weighted by category frequency: decision correct "
                  f"{100 * num / den:5.1f}%, explanation correct {100 * num_e / den:5.1f}%")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"details -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", default=str(ROOT / "data" / "raw" / "eval_patient_doctor_encounters.jsonl"))
    parser.add_argument("--decisions", default=str(ROOT / "submissions" / "final" / "eval_decisions.jsonl"))
    parser.add_argument("--tags", type=int, default=80)
    parser.add_argument("--vetoes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "explanations_check.jsonl"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
