"""Add our high-precision rule spans (ages, dates, times, contacts, IDs) to teacher-labelled real encounters.

The teacher misses some formats our rules define by policy ("at age 7", seasons, weekdays); without them
the student would learn to leave those untagged. Only spans from rules (not from spaCy NER) are added,
and only where the teacher has nothing overlapping.

    uv run python scripts/merge_rule_spans.py data/distill/teacher_labels_heldout.jsonl \\
        <decisions of the rule-based pipeline on those encounters> data/distill/train_real_heldout.jsonl
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from clinanon.io import read_jsonl, write_jsonl

RULE_CATEGORIES = {"AGE", "DATE", "TIME", "CONTACT", "IDENTIFIER"}


def main(labels_path: str, decisions_path: str, out_path: str) -> None:
    rules = defaultdict(list)
    for d in read_jsonl(decisions_path):
        if d["action"] == "tag" and d["category"] in RULE_CATEGORIES and d["source"].startswith("rule:"):
            rules[d["encounter_id"]].append(d)
    added = Counter()
    records = read_jsonl(labels_path)
    for r in records:
        for d in rules[r["encounter_id"]]:
            if r[d["field"]][d["start"] : d["end"]] != d["text"]:
                continue  # offsets refer to the same normalized text; skip if not
            if any(s["field"] == d["field"] and s["start"] < d["end"] and d["start"] < s["end"] for s in r["spans"]):
                continue
            r["spans"].append({"field": d["field"], "start": d["start"], "end": d["end"], "text": d["text"],
                               "category": d["category"]})
            added[d["category"]] += 1
    write_jsonl(out_path, records)
    print(f"added rule spans: {dict(added)} -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
