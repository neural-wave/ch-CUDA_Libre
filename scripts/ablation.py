"""Ablation: switch off one component at a time and score privacy and utility on every data set.

Privacy: share of private information removed (gold names/ages on ACI, the Presidio-strict judge on ACI,
known values on the synthetic test, silver values on the robustness test). Utility: medical concept
recall (clinical-vocabulary tokens and ICD-description terms still present).

    uv run python scripts/ablation.py --variant default        # one variant -> outputs/ablation/default.json
    uv run python scripts/ablation.py --table                  # summary of all finished variants
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest import mock

from clinanon import propagation, veto
from clinanon.detectors import rules
from clinanon.evaluation.gold import load_gold
from clinanon.evaluation.judges import PresidioJudge, _presidio
from clinanon.evaluation.metrics import gold_privacy, judge_privacy, utility
from clinanon.io import read_jsonl
from clinanon.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_synthetic import coverage  # noqa: E402

DATA = {
    "aci_eval": (ROOT / "data/raw/eval_patient_doctor_encounters.jsonl", None),
    "aci_other": (ROOT / "data/raw/patient_doctor_encounters.jsonl", None),
    "synthetic_test": (ROOT / "data/synthetic/synthetic_test.jsonl", ROOT / "data/synthetic/synthetic_test_gold.jsonl"),
    "robustness_test": (ROOT / "data/robustness/robustness_test.jsonl", ROOT / "data/robustness/robustness_test_gold.jsonl"),
}
_ALL = (lambda spans, texts, protect: (spans, []))  # noqa: E731

VARIANTS = {  # name -> (Pipeline.from_config kwargs, patches)
    "rules_spacy": ({"student_model": None}, {}),
    "rules_only": ({"student_model": None, "use_ner": False}, {}),
    "spacy_only": ({"student_model": None}, {(rules, "detect"): lambda text, field: []}),
    "no_veto": ({"student_model": None}, {(veto, "apply"): _ALL}),
    "no_propagation": ({"student_model": None}, {(propagation, "find_mentions"): lambda texts, names: [],
                                                 (propagation, "find_place_mentions"): lambda texts, places: []}),
    "no_truecasing": ({"student_model": None, "use_truecasing": False}, {}),
    "default": ({}, {}),
    "rules_student": ({"use_ner": False}, {}),
    "default_no_veto": ({}, {(veto, "apply"): _ALL}),
}


class PresidioOffTheShelf:
    """Presidio's default analyzer, every detection replaced by a tag: no rules, veto or propagation."""

    def anonymize(self, record: dict) -> tuple[dict, list]:
        out, tagged = dict(record), []
        for field in ("note", "dialogue"):
            text = record[field]
            results = sorted(_presidio().analyze(text=text, language="en"), key=lambda r: (-r.score, r.start))
            chosen = []
            for r in results:
                if not any(r.start < c.end and c.start < r.end for c in chosen):
                    chosen.append(r)
            for r in sorted(chosen, key=lambda r: r.start, reverse=True):
                text = text[: r.start] + f"[{r.entity_type}]" + text[r.end :]
                tagged.append((record["encounter_id"], field, r.start, r.end))
            out[field] = text
        return out, tagged


def _run(pipeline, records: list[dict]) -> tuple[list[dict], dict, float]:
    outputs, tagged = [], defaultdict(list)
    start = time.perf_counter()
    for record in records:
        out, decisions = pipeline.anonymize(record)
        outputs.append(out)
        for d in decisions:
            if isinstance(d, tuple):
                tagged[(d[0], d[1])].append((d[2], d[3]))
            elif d.action == "tag":
                tagged[(d.encounter_id, d.field)].append((d.start, d.end))
    return outputs, tagged, (time.perf_counter() - start) / len(records)


def score(name: str) -> dict:
    if name == "presidio":
        pipeline, patches = PresidioOffTheShelf(), {}
    else:
        kwargs, patches = VARIANTS[name]
        pipeline = Pipeline.from_config(**kwargs)
    judge, gold = PresidioJudge(strict=True), load_gold()
    result = {}
    with contextlib.ExitStack() as stack:
        for (module, attr), fn in patches.items():
            stack.enter_context(mock.patch.object(module, attr, fn))
        for data, (path, gold_path) in DATA.items():
            records = read_jsonl(path)
            outputs, tagged, seconds = _run(pipeline, records)
            row = {"documents": len(records), "seconds_per_document": round(seconds, 3), **utility(records, outputs)}
            if gold_path is None:
                row["gold_names_ages_removed_%"] = gold_privacy(records, outputs, gold)["overall_removed_%"]
                row["judge_removed_%"] = judge_privacy(records, outputs, judge)["removed_%"]
                row["tags"] = sum(map(len, tagged.values()))
            else:
                values = read_jsonl(gold_path)
                full = sum(coverage(v, tagged[(v["encounter_id"], v["field"])]) == 1.0 for v in values)
                row["values"], row["values_removed_%"] = len(values), round(100 * full / len(values), 1)
            result[data] = row
            print(name, data, row, flush=True)
    return result


def table(out: Path) -> None:
    cols = [("aci_other", "gold_names_ages_removed_%"), ("aci_other", "judge_removed_%"),
            ("aci_eval", "gold_names_ages_removed_%"), ("aci_eval", "judge_removed_%"),
            ("synthetic_test", "values_removed_%"), ("robustness_test", "values_removed_%"),
            ("aci_other", "clinical_token_retention_%"), ("aci_other", "clinical_tokens_lost"),
            ("aci_eval", "clinical_token_retention_%"), ("aci_eval", "clinical_tokens_lost"),
            ("aci_other", "icd_term_retention_%"), ("aci_eval", "icd_term_retention_%")]
    print("variant".ljust(16) + "".join(f"{d[:9]}:{m[:10]}".rjust(22) for d, m in cols))
    for path in sorted(out.glob("*.json")):
        r = json.loads(path.read_text())
        print(path.stem.ljust(16) + "".join(str(r[d].get(m)).rjust(22) for d, m in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=[*VARIANTS, "presidio"])
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "ablation"))
    args = parser.parse_args()
    out = Path(args.out)
    if args.table:
        table(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.variant}.json").write_text(json.dumps(score(args.variant), indent=1))


if __name__ == "__main__":
    main()
