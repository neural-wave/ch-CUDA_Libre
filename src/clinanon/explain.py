"""Explanations for every anonymization decision (bonus task: explainability).

Each decision in ``decisions.jsonl`` gets a one-sentence ``explanation``: what was found, by which evidence
(rule, gazetteer, statistical model with its confidence), why it is private (taxonomy category and its
literature source) and what was done. Rejected candidates are explained too ("followed by a clinical
head noun, so it names a clinical concept"). ``render_html`` turns an encounter and its decisions into a page where every highlighted span
shows its explanation on hover.
"""

from __future__ import annotations

import html
from collections import defaultdict

from clinanon.taxonomy import Taxonomy
from clinanon.types import Decision, Span

# Evidence behind each detector, keyed by the source prefix ("rule:title" -> "rule:title").
_EVIDENCE = {
    "rule:title": "a title in front of it (Dr., Mr., Ms., doctor)",
    "rule:dictation_header": "a dictation header naming the patient",
    "rule:subject_with_age": "it is the subject of a sentence stating an age",
    "rule:my_name_is": "a self-introduction",
    "rule:vocative": "it is used to address someone after a greeting or thanks",
    "rule:turn_start_address": "the doctor addresses the patient with it at the start of a turn",
    "rule:relative_named": "it names a relative or another person in apposition",
    "rule:age": "an age expression",
    "rule:age_number": "a number stated as someone's age",
    "rule:name_comma_age": "an age in apposition to a name",
    "rule:time_anchor": "a visit-relative time word",
    "rule:labeled_id": "a number after a label (MRN, insurance, policy, AHV …)",
    "rule:long_number": "a number too long to be a measurement",
    "rule:grouped_number": "a grouped number too long to be a measurement",
    "rule:alphanumeric_id": "a letter-and-digit identifier",
    "rule:swiss_ahv": "the Swiss AHV number format",
    "rule:ssn": "the US social security number format",
    "rule:glued_mrn": "a record number glued to its label",
    "rule:phone": "a phone number format",
    "rule:email": "an e-mail address (written or spoken)",
    "rule:us_street": "a street address", "rule:eu_street": "a street address",
    "rule:german_street": "a street address", "rule:romance_street": "a street address",
    "rule:numbered_street": "a street address", "rule:street_name": "a named street",
    "rule:zip_with_city": "a postal code", "rule:us_state_zip": "a state and ZIP code",
    "rule:uk_postcode": "a UK postcode", "rule:mountain": "a named mountain",
    "rule:facility": "a proper name in front of hospital, clinic or center",
    "rule:foreign_facility": "a named hospital or clinic (Spital, Klinik, Ospedale, Hôpital …)",
    "rule:org_cue": "it follows a phrase introducing an employer or insurer",
    "rule:occupation": "a job title after \"is a\", \"works as\" or \"retired\"",
    "rule:job_cue": "it follows \"works as\" or an occupation label",
    "rule:owns_business": "a business the person owns",
    "rule:assistant_name": "the name of a documentation assistant or patient portal",
    "gazetteer:geonames": "a city, region or country listed in GeoNames, after a word introducing a place or a postcode",
    "propagation": "the same name or place was identified elsewhere in this encounter",
    "ner:student": "the distilled student model (trained on teacher-labelled clinical text)",
    "ner:gliner": "the GLiNER zero-shot model",
    "ner": "spaCy named-entity recognition",
    "pattern": "a Presidio pattern recognizer",
}


# Rules not listed above, by the category they produce.
_RULE_BY_CATEGORY = {
    "DATE": "a calendar pattern (date, month, year, weekday, holiday or season)",
    "TIME": "a clock-time pattern", "AGE": "an age pattern", "LOCATION": "an address or place pattern",
    "IDENTIFIER": "an identifier pattern", "CONTACT": "a contact-detail pattern",
}


def evidence(source: str, category: str = "") -> str:
    for key in sorted(_EVIDENCE, key=len, reverse=True):
        if source == key or source.startswith(key + "-") or source.startswith(key + ":") or (
            key.startswith("rule:") and source.startswith(key)):
            return _EVIDENCE[key]
    if source.startswith("rule:") and category in _RULE_BY_CATEGORY:
        return _RULE_BY_CATEGORY[category]
    return source


def explain(span: Span, action: str, replacement: str | None, taxonomy: Taxonomy) -> str:
    cat = taxonomy.categories[span.category]
    found = f"\"{span.text}\" is {cat.label} ({cat.identifier} identifier; {', '.join(cat.sources)})"
    how = evidence(span.source, span.category)
    if span.source.startswith(("ner", "pattern")) and span.score < 1.0:
        how += f", confidence {span.score:.2f}"
    if action == "tag":
        return f"{found}; evidence: {how}; replaced by {replacement}."
    return f"{found}; evidence: {how}; kept by policy: {cat.why_kept or 'not identifying'}."


def explain_veto(span: Span, reason: str, removed_by: str | None = None) -> str:
    """A rejected detection. ``removed_by``: the evidence of another detection that removes the span anyway."""
    text = (f"Candidate rejected: \"{span.text}\" was proposed as {span.category.lower()} (evidence: "
            f"{evidence(span.source, span.category)}), but {reason}.")
    if removed_by:
        text += f" It is removed anyway (evidence: {removed_by})."
    return text


# ----------------------------------------------------------------------------------------------- HTML

_CSS = """
body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1d1d1f}
header{padding:14px 20px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:22px 0 6px}
.legend span{margin-right:14px}.enc{background:#fff;margin:16px 20px;padding:12px 18px;border:1px solid #ddd;border-radius:8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.cols{grid-template-columns:1fr}}
pre{white-space:pre-wrap;word-wrap:break-word;font:13px/1.55 ui-monospace,Menlo,monospace;margin:0}
.tag{background:#ffd6d6;border-bottom:2px solid #d93025;cursor:help}
.tag sup{font-size:10px;color:#a50e0e;font-weight:600;margin-left:2px}
.keep{background:#d9f2e3;border-bottom:2px dotted #188038;cursor:help}
.policy{background:#e3ecfd;border-bottom:2px dotted #1a56db;cursor:help}
.stats{color:#555;font-size:13px}
"""


def render_html(records: list[dict], decisions: list[Decision], title: str = "clinanon decisions") -> str:
    """One section per encounter: the original text with every decision highlighted and explained."""
    by_record = defaultdict(list)
    for d in decisions:
        by_record[(d.encounter_id, d.field)].append(d)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style>"
             f"</head><body><header><h1>{html.escape(title)}</h1><div class='legend'>"
             "<span class='tag'>removed</span><span class='keep'>clinical look-alike kept</span>"
             "<span class='policy'>kept by policy (relative time)</span> — hover a span for its explanation"
             "</div></header>"]
    for r in records:
        eid = r["encounter_id"]
        n_tag = sum(d.action == "tag" for f in ("dialogue", "note") for d in by_record[(eid, f)])
        n_keep = sum(d.action != "tag" for f in ("dialogue", "note") for d in by_record[(eid, f)])
        parts.append(f"<section class='enc'><h2>{html.escape(eid)}</h2><div class='stats'>{n_tag} removed, "
                     f"{n_keep} kept with a reason</div><div class='cols'>")
        for field in ("dialogue", "note"):
            parts.append(f"<div><h2>{field}</h2><pre>{_highlight(r[field], by_record[(eid, field)])}</pre></div>")
        parts.append("</div></section>")
    parts.append("</body></html>")
    return "".join(parts)


def _highlight(text: str, decisions: list[Decision]) -> str:
    # Removed spans first (they never overlap); then kept spans that do not overlap anything shown.
    chosen: list[Decision] = []
    for d in sorted(decisions, key=lambda d: (d.action != "tag", d.start, -(d.end - d.start))):
        if all(d.end <= c.start or c.end <= d.start for c in chosen):
            chosen.append(d)
    out, pos = [], 0
    for d in sorted(chosen, key=lambda d: d.start):
        out.append(html.escape(text[pos : d.start]))
        css = {"tag": "tag", "veto": "keep"}.get(d.action, "policy")
        badge = f"<sup>{html.escape(d.replacement)}</sup>" if d.action == "tag" and d.replacement else ""
        out.append(f"<span class='{css}' title='{html.escape(d.explanation or d.reason, quote=True)}'>"
                   f"{html.escape(text[d.start : d.end])}{badge}</span>")
        pos = d.end
    out.append(html.escape(text[pos:]))
    return "".join(out)
