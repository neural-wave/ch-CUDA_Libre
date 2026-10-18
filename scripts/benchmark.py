"""Efficiency benchmark (bonus task): runtime, peak RAM and VRAM per pipeline profile, next to its quality.

Each profile runs in a fresh process (so peak RAM is its own) on the 67 ACI eval encounters and the 40
synthetic test encounters:

    fast      rules only                      (no statistical model)
    balanced  rules + distilled student       (no spaCy)
    spacy     rules + spaCy                   (no student)
    default   rules + spaCy + student         (the submission)

    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "fast": {"use_ner": False, "student_model": None},
    "balanced": {"use_ner": False, "student_model": "auto"},
    "spacy": {"use_ner": True, "student_model": None},
    "default": {"use_ner": True, "student_model": "auto"},
}

WORKER = r'''
import json, resource, sys, time
options = json.loads(sys.argv[1])
t0 = time.perf_counter()
from clinanon.pipeline import Pipeline
from clinanon.io import read_jsonl
from clinanon.evaluation import metrics
from clinanon.evaluation.gold import load_gold
pipeline = Pipeline.from_config(**options)
load = time.perf_counter() - t0
eval_records = read_jsonl("data/raw/eval_patient_doctor_encounters.jsonl")
t1 = time.perf_counter()
outputs, _ = pipeline.anonymize_all(eval_records)
per_encounter = (time.perf_counter() - t1) / len(eval_records)
gold = metrics.gold_privacy(eval_records, outputs, load_gold())["overall_removed_%"]
utility = metrics.utility(eval_records, outputs)
synth = read_jsonl("data/synthetic/synthetic_test.jsonl")
synth_gold = read_jsonl("data/synthetic/synthetic_test_gold.jsonl")
_, decisions = pipeline.anonymize_all(synth)
tagged = {}
for d in decisions:
    if d.action == "tag":
        tagged.setdefault((d.encounter_id, d.field), []).append((d.start, d.end))
removed = 0
for g in synth_gold:
    chars = [i for i in range(g["start"], g["end"]) if g["text"][i - g["start"]].isalnum()]
    spans = tagged.get((g["encounter_id"], g["field"]), [])
    removed += all(any(s <= i < e for s, e in spans) for i in chars)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux
try:
    import torch
    vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
except ImportError:
    vram = 0.0
print(json.dumps({"load_s": load, "s_per_encounter": per_encounter, "peak_ram_mb": peak_mb, "vram_mb": vram,
                  "gold_names_ages_removed_%": gold, "synthetic_test_removed_%": 100 * removed / len(synth_gold),
                  "clinical_token_retention_%": utility["clinical_token_retention_%"]}))
'''


def main() -> None:
    results = {}
    print(f"{'profile':10s} {'load s':>7s} {'s/enc':>6s} {'peak RAM':>9s} {'VRAM':>5s} {'gold':>6s} "
          f"{'synthetic':>9s} {'clinical':>8s}")
    for name, options in PROFILES.items():
        proc = subprocess.run([sys.executable, "-c", WORKER, json.dumps(options)], cwd=ROOT, capture_output=True,
                              text=True, check=True)
        r = json.loads(proc.stdout.strip().splitlines()[-1])
        results[name] = r
        print(f"{name:10s} {r['load_s']:7.1f} {r['s_per_encounter']:6.2f} {r['peak_ram_mb']:7.0f}MB "
              f"{r['vram_mb']:5.0f} {r['gold_names_ages_removed_%']:5.1f}% {r['synthetic_test_removed_%']:8.1f}% "
              f"{r['clinical_token_retention_%']:7.1f}%")
    out = ROOT / "outputs" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
