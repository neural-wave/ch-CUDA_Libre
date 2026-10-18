"""Evaluation harness: privacy (gold + judge proxies), utility proxies and the damage audit."""

from __future__ import annotations

from pathlib import Path

from clinanon.evaluation import damage, metrics
from clinanon.evaluation.gold import load_gold
from clinanon.evaluation.judges import default_judges


def evaluate(originals: list[dict], anonymized: list[dict], audit_csv: str | Path | None = None) -> dict:
    """Score ``anonymized`` against ``originals`` (both lists of encounters, matched by encounter_id)."""
    by_id = {r["encounter_id"]: r for r in anonymized}
    missing = [r["encounter_id"] for r in originals if r["encounter_id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} encounters missing from the anonymized file, e.g. {missing[:3]}")
    anonymized = [by_id[r["encounter_id"]] for r in originals]
    codes_changed = sum(o["icd10_codes"] != a["icd10_codes"] for o, a in zip(originals, anonymized, strict=True))

    gold = load_gold()
    edits = damage.audit(originals, anonymized, gold)
    if audit_csv:
        damage.write_csv(edits, audit_csv)
    return {
        "encounters": len(originals),
        "icd10_codes_changed": codes_changed,
        "privacy_gold": metrics.gold_privacy(originals, anonymized, gold),
        "privacy_judges": [metrics.judge_privacy(originals, anonymized, j) for j in default_judges()],
        "utility": metrics.utility(originals, anonymized),
        "damage": damage.summarize(edits),
    }


def format_report(report: dict) -> str:
    """Compact human-readable summary of an evaluation report."""
    lines = [f"encounters: {report['encounters']}   icd10_codes changed: {report['icd10_codes_changed']}"]
    gold = report["privacy_gold"]
    lines.append(f"\nPRIVACY - ACI metadata gold: {gold['overall_removed_%']}% of mentions removed")
    for kind, row in gold["by_kind"].items():
        lines.append(f"  {kind:20s} {row['mentions']:4d} mentions  removed {row['removed_%']}%")
    for j in report["privacy_judges"]:
        lines.append(
            f"\nPRIVACY - judge {j['judge']}: {j['removed_%']}% of {j['detected_in_original']} detected values "
            f"removed; {j['residual_detections_in_output']} detections left in output"
        )
        for label, row in list(j["by_label"].items())[:8]:
            lines.append(f"  {label:20s} {row['detected']:5d} detected  removed {row['removed_%']}%")
        lines.append("  top survivors: " + ", ".join(f"{v} ({n})" for v, n in j["top_survivors"][:12]))
    u = report["utility"]
    lines.append(
        f"\nUTILITY - ICD-term retention {u['icd_term_retention_%']}%, clinical-token retention "
        f"{u['clinical_token_retention_%']}% ({u['clinical_tokens_lost']} clinical tokens lost)"
    )
    d = report["damage"]
    lines.append(f"\nDAMAGE AUDIT - {d['edits']} edits, {d['suspected_clinical']} look clinical:")
    lines.extend(f"  {n:3d}x  {edit}" for edit, n in d["top_suspected"][:20])
    return "\n".join(lines)
