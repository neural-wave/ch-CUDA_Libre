"""Recall per type of private detail on the synthetic encounters (see scripts/make_synthetic.py).

A gold value counts as removed when every one of its letters and digits lies inside a tagged
span, and as partly removed when only some do ("+41 79 [CONTACT]" still leaks part of a number).

    uv run python scripts/eval_synthetic.py --split dev
    uv run --group judges python scripts/eval_synthetic.py --split dev --gliner nvidia/gliner-PII
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from clinanon.evaluation.metrics import utility
from clinanon.io import read_jsonl, write_jsonl
from clinanon.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def coverage(item: dict, tagged: list[tuple[int, int]]) -> float:
    chars = [i for i in range(item["start"], item["end"]) if item["text"][i - item["start"]].isalnum()]
    covered = sum(any(s <= i < e for s, e in tagged) for i in chars)
    return covered / len(chars) if chars else 1.0


def style(item: dict, dataset: str) -> str:
    if item["field"] == "note":
        return "note"
    return "asr" if dataset.endswith("-asr") or dataset == "synthetic-asr" else "cased"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", help="dev, test, or a name for --records/--gold")
    parser.add_argument("--records", default=None, help="other records to score (default: the synthetic split)")
    parser.add_argument("--gold", default=None, help="gold values for --records")
    parser.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    parser.add_argument("--gliner", default=None, help="also run this GLiNER model as a detector")
    parser.add_argument("--student", default=None, help="also run this distilled student model ('auto': models/student)")
    parser.add_argument("--no-spacy", action="store_true", help="do not run the spaCy/Presidio NER")
    parser.add_argument("--name", default=None, help="name of this run in the output files")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "synthetic"))
    args = parser.parse_args()

    records = read_jsonl(args.records or Path(args.data) / f"synthetic_{args.split}.jsonl")
    gold = read_jsonl(args.gold or Path(args.data) / f"synthetic_{args.split}_gold.jsonl")
    pipeline = Pipeline.from_config(gliner_model=args.gliner, student_model=args.student, use_ner=not args.no_spacy)
    start = time.perf_counter()
    outputs, decisions = pipeline.anonymize_all(records)
    elapsed = time.perf_counter() - start

    tagged: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for d in decisions:
        if d.action == "tag":
            tagged[(d.encounter_id, d.field)].append((d.start, d.end))
    datasets = {r["encounter_id"]: r["dataset"] for r in records}
    texts = {(r["encounter_id"], f): r[f] for r in records for f in ("note", "dialogue")}

    rows, misses = defaultdict(lambda: [0, 0, 0]), []  # (type, style) -> [values, removed, partly removed]
    for item in gold:
        c = coverage(item, tagged[(item["encounter_id"], item["field"])])
        for key in ((item["type"], style(item, datasets[item["encounter_id"]])), (item["type"], "all"),
                    ("ALL", style(item, datasets[item["encounter_id"]])), ("ALL", "all")):
            rows[key][0] += 1
            rows[key][1] += c == 1.0
            rows[key][2] += 0 < c < 1.0
        if c < 1.0:
            text = texts[(item["encounter_id"], item["field"])]
            misses.append({**item, "coverage": round(c, 2), "style": style(item, datasets[item["encounter_id"]]),
                           "context": text[max(0, item["start"] - 60) : item["end"] + 40].replace("\n", " | ")})

    name = f"{args.split}_{args.name}" if args.name else f"{args.split}{'_' + args.gliner.split('/')[-1] if args.gliner else ''}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"{name}_anonymized.jsonl", outputs)
    with (out / f"{name}_misses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["encounter_id", "field", "style", "type", "category", "text",
                                               "coverage", "context"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(misses)

    styles = ["asr", "cased", "note", "all"]
    print(f"{args.split}: {len(records)} records, {len(gold)} gold values, {elapsed / len(records):.2f} s/record"
          f"{' (+ ' + args.gliner + ')' if args.gliner else ''}")
    print(f"{'removed % (partly %)':24s}" + "".join(f"{s:>22s}" for s in styles))
    summary = {}
    for kind in sorted({k for k, _ in rows}, key=lambda k: (k == "ALL", k)):
        cells = []
        for s in styles:
            n, full, part = rows.get((kind, s), (0, 0, 0))
            cells.append(f"{100 * full / n:5.1f} ({100 * part / n:4.1f}) n={n:<4d}" if n else "-")
            summary[f"{kind}/{s}"] = {"values": n, "removed_%": round(100 * full / n, 1) if n else None,
                                      "partly_%": round(100 * part / n, 1) if n else None}
        print(f"{kind:24s}" + "".join(f"{c:>22s}" for c in cells))
    u = utility(records, outputs)
    print(f"utility on the synthetic records: ICD-term retention {u['icd_term_retention_%']}%, "
          f"clinical-token retention {u['clinical_token_retention_%']}%")
    (out / f"{name}_report.json").write_text(json.dumps({"summary": summary, "utility": u,
                                                         "seconds_per_record": elapsed / len(records)}, indent=2))
    print(f"misses -> {out / (name + '_misses.csv')}")


if __name__ == "__main__":
    main()
