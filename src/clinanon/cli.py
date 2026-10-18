"""Command-line entry point.

    clinanon anonymize -i data/raw/eval_patient_doctor_encounters.jsonl -o outputs/eval_anon.jsonl
    clinanon evaluate  -i data/raw/eval_patient_doctor_encounters.jsonl -a outputs/eval_anon.jsonl
    clinanon explain   -i data/raw/eval_patient_doctor_encounters.jsonl -d outputs/eval_decisions.jsonl \
                       -o outputs/eval_explained.html
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from clinanon.io import read_jsonl, write_jsonl

log = logging.getLogger("clinanon")


_WORKER_PIPELINE = None


def _pipeline_options(args: argparse.Namespace) -> dict:
    return {"taxonomy_path": args.taxonomy, "use_ner": not args.no_spacy, "use_truecasing": not args.no_truecasing,
            "normalize_asr": not args.keep_asr_splits, "gliner_model": args.gliner,
            "student_model": args.student, "use_icd": not args.no_icd}


def _init_worker(options: dict) -> None:
    global _WORKER_PIPELINE
    import torch

    from clinanon.pipeline import Pipeline

    torch.set_num_threads(1)  # one thread per worker process; parallelism comes from the processes
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    _WORKER_PIPELINE = Pipeline.from_config(**options)


def _anonymize_chunk(records: list[dict]):
    return _WORKER_PIPELINE.anonymize_all(records)


def _anonymize(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    start = time.perf_counter()
    if args.method == "identity":
        # Reference point for the evaluation: no anonymization at all.
        outputs, decisions = records, []
    elif args.workers > 1:
        # Encounters are independent: split them over worker processes (each loads its own models).
        from concurrent.futures import ProcessPoolExecutor

        size = max(1, len(records) // (4 * args.workers))
        parts = [records[i : i + size] for i in range(0, len(records), size)]
        outputs, decisions = [], []
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(_pipeline_options(args),)) as pool:
            for out, dec in pool.map(_anonymize_chunk, parts):
                outputs.extend(out)
                decisions.extend(dec)
    else:
        from clinanon.pipeline import Pipeline  # heavy import (spaCy), only when needed

        pipeline = Pipeline.from_config(**_pipeline_options(args))
        outputs, decisions = pipeline.anonymize_all(records)
    elapsed = time.perf_counter() - start
    write_jsonl(args.output, outputs)
    if args.decisions:
        write_jsonl(args.decisions, (d.to_dict() for d in decisions))
    log.info("anonymized %d encounters in %.1fs (%.2fs/encounter) -> %s",
             len(records), elapsed, elapsed / max(len(records), 1), args.output)


def _evaluate(args: argparse.Namespace) -> None:
    from clinanon.evaluation import evaluate, format_report

    report = evaluate(read_jsonl(args.input), read_jsonl(args.anonymized), audit_csv=args.audit)
    print(format_report(report))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _explain(args: argparse.Namespace) -> None:
    from clinanon.explain import render_html
    from clinanon.pipeline import normalize_asr_splits
    from clinanon.types import TEXT_FIELDS, Decision

    records = read_jsonl(args.input)
    if args.ids:
        records = [r for r in records if r["encounter_id"] in set(args.ids)]
    records = records[: args.limit or None]
    # decision offsets refer to the text after the ASR repair ("wan na" -> "wanna")
    records = [{**r, **{f: normalize_asr_splits(r[f]) for f in TEXT_FIELDS}} for r in records]
    ids = {r["encounter_id"] for r in records}
    decisions = [Decision(**d) for d in read_jsonl(args.decisions) if d["encounter_id"] in ids]
    explained = sum(bool(d.explanation or d.reason) for d in decisions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(render_html(records, decisions), encoding="utf-8")
    log.info("%d encounters, %d decisions, %.1f%% with an explanation -> %s", len(records), len(decisions),
             100 * explained / max(len(decisions), 1), args.output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clinanon", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("anonymize", help="anonymize a JSONL file of encounters")
    p.add_argument("-i", "--input", required=True, help="input JSONL (dialogue, note, icd10_codes, ...)")
    p.add_argument("-o", "--output", required=True, help="output JSONL, same schema as the input")
    p.add_argument("-d", "--decisions", help="optional JSONL log of every anonymization decision")
    p.add_argument("--method", choices=["pipeline", "identity"], default="pipeline")
    p.add_argument("--taxonomy", default=None, help="privacy taxonomy YAML (default: configs/)")
    p.add_argument("--no-truecasing", action="store_true", help="run NER on lower-cased dialogues as they are")
    p.add_argument("--keep-asr-splits", action="store_true",
                   help="do not rejoin ASR-split contractions ('wan na' -> 'wanna')")
    p.add_argument("--student", default="auto", metavar="DIR",
                   help="distilled student model directory (default: models/student if present)")
    p.add_argument("--no-student", dest="student", action="store_const", const=None,
                   help="do not run the distilled student")
    p.add_argument("--no-spacy", action="store_true", help="do not run the spaCy/Presidio NER")
    p.add_argument("--workers", type=int, default=1, help="process encounters in parallel worker processes")
    p.add_argument("--no-icd", action="store_true",
                   help="do not use the icd10_codes at inference (bonus task); the field is still copied")
    p.add_argument("--gliner", default=None, metavar="MODEL",
                   help="also run a GLiNER PII model as detector, e.g. nvidia/gliner-PII (needs the judges group)")
    p.set_defaults(func=_anonymize)

    p = sub.add_parser("evaluate", help="score an anonymized file against the original")
    p.add_argument("-i", "--input", required=True, help="original JSONL")
    p.add_argument("-a", "--anonymized", required=True, help="anonymized JSONL")
    p.add_argument("--audit", help="write the damage audit (every edit) to this CSV")
    p.add_argument("--report", help="write the full report to this JSON file")
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("explain", help="HTML view of the decisions, with an explanation on every span")
    p.add_argument("-i", "--input", required=True, help="original JSONL")
    p.add_argument("-d", "--decisions", required=True, help="decisions JSONL written by `anonymize -d`")
    p.add_argument("-o", "--output", required=True, help="HTML file to write")
    p.add_argument("--ids", nargs="*", help="only these encounter ids")
    p.add_argument("--limit", type=int, default=0, help="only the first N encounters")
    p.set_defaults(func=_explain)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # very chatty at INFO/WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per file of the model download
    args.func(args)


if __name__ == "__main__":
    main()
