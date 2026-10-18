"""Find the stand-in judge that best reproduces the official mid-term privacy score.

We sent one anonymized file and got one number back (privacy removal proportion 0.9179). Every candidate
judge scores the same file under two readings of "proportion removed":

  value   - share of the values the judge detects in the original that no longer appear in the output
  count   - 1 - (detections in output / detections in original)

A judge/reading that lands close to the official number is a plausible model of the organizers' scorer,
and becomes our development metric. One number cannot prove which scorer they use, but it rules many out.

Usage:
    uv run --group judges python scripts/calibrate_judges.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from clinanon.evaluation import metrics
from clinanon.evaluation.judges import candidate_judges
from clinanon.io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MIDTERM = 0.9179


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", default=ROOT / "data/raw/eval_patient_doctor_encounters.jsonl")
    parser.add_argument(
        "--anonymized", default=ROOT / "submissions/midterm/eval_patient_doctor_encounters_anonymized.jsonl"
    )
    parser.add_argument("--target", type=float, default=OFFICIAL_MIDTERM)
    parser.add_argument("--only", nargs="*", help="judge names to run (default: all candidates)")
    parser.add_argument("--out", default=ROOT / "outputs/calibration")
    args = parser.parse_args()
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    originals, anonymized = read_jsonl(args.original), read_jsonl(args.anonymized)
    judges = candidate_judges()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for judge in judges:
        if args.only and judge.name not in args.only:
            continue
        started = time.time()
        result = metrics.judge_privacy(originals, anonymized, judge)
        result["seconds"] = round(time.time() - started, 1)
        (out_dir / f"{judge.name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{judge.name:32s} done in {result['seconds']}s", flush=True)

    print(summary_table(out_dir, args.target))


def summary_table(out_dir: Path, target: float) -> str:
    rows = []
    for path in sorted(out_dir.glob("*.json")):
        r = json.loads(path.read_text())
        for reading, key in (("value", "removed_%"), ("count", "count_removed_%")):
            if r.get(key) is not None:
                rows.append((abs(r[key] / 100 - target), r["judge"], reading, r[key], r["detected_in_original"]))
    lines = [f"\n{'judge':32s} {'reading':7s} {'removed':>8s} {'|diff|':>7s} {'detected':>9s}   (target {target})"]
    for diff, name, reading, value, detected in sorted(rows):
        lines.append(f"{name:32s} {reading:7s} {value:7.1f}% {100 * diff:6.1f}  {detected:9d}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
