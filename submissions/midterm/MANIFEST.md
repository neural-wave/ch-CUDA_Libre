# Mid-term evaluation submission

**File to send:** `eval_patient_doctor_encounters_anonymized.jsonl` (67 encounters, 635184 bytes)
**SHA-256:** `cb96420af81bdad1b9a6b6b902f6f094a31413e17a9c944a505af19b87494275`
**Generated:** 2026-10-17 16:28 with clinanon v0.1.0 (pipeline v1)

Kept for reference: the final submission is in `submissions/final/`.

## Configuration

```
clinanon anonymize -i data/raw/eval_patient_doctor_encounters.jsonl \
    -o submissions/midterm/eval_patient_doctor_encounters_anonymized.jsonl
```

- Default taxonomy (`configs/privacy_taxonomy.yaml`): every category is replaced by a typed tag.
- **TIME_ANCHOR = keep**: "today", "yesterday", weekdays and holidays are left in the text.
- Durations and frequencies ("two weeks", "daily") are never touched.
- ASR contraction repair on ("wan na" -> "wanna", "lem me" -> "lemme").
- Rules + spaCy NER (en_core_web_lg) with note-guided truecasing. No GLiNER, no LLM.

## Local predictions (to compare with the official numbers)

| Metric | Local value |
|---|---|
| Gold patient names/ages removed (ACI metadata) | 100.0 % |
| Privacy, judge = Presidio default (counts "today", "daily", "C7"...) | 31.0 % |
| Privacy, judge = Presidio strict (no relative time, no noisy number recognizers) | 83.6 % |
| ICD-description term retention | 100.0 % |
| Clinical-token retention | 99.8 % |

## How to read the official privacy score

- **Around 30 %:** their detector counts relative time expressions ("today", "daily") as PII. Decide
  whether to tag TIME_ANCHOR (one-line change in the taxonomy) and possibly durations, weighing the
  utility cost.
- **Around 80-90 %:** their detector behaves like a sensible PII detector. Keep the current policy and
  spend effort on recall for names, places and organizations.
- **Well above 90 %:** they probably score against annotated PII spans rather than raw detector hits.
  Focus on residual name/location recall.

## Official numbers (received 2026-10-17)

| Metric | Value |
|---|---|
| Privacy removal proportion | 0.9179 |
| ICD-10 F1 delta (original vs anonymized input to their predictor) | 0.0 |
| Medical concept recall | 0.9996 |

Reading: a detector that counts relative time would have given about 0.31, so TIME_ANCHOR = keep was
not penalized. Utility is at its ceiling, so there is room to remove more. The remaining 8 % is the
target of plan A (calibrate a stand-in judge to 0.918).
