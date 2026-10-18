# Final submission

The deliverable is `eval_patient_doctor_encounters_anonymized.jsonl`:
- 67 encounters, 635,027 bytes
- SHA-256 `7349eee016b0d3ab0c552c7515113f053332aa874ec737e6bc5d92f251a521c6`

## Files

| File | Content |
|---|---|
| `eval_patient_doctor_encounters_anonymized.jsonl` | Anonymized `data/raw/eval_patient_doctor_encounters.jsonl` with the same schema |
| `eval_decisions.jsonl` | 1,377 decisions (removed, kept by policy, rejected candidates), each with an explanation. SHA-256 `d4f540952f736ff8de26b20690d2393a5448a0cebc805273e5735bf65ae92f6d` |
| `eval_explained.html` | The same decisions, highlighted in the original text; hover a span for its explanation |

## How it was produced

```bash
python -m clinanon anonymize -i data/raw/eval_patient_doctor_encounters.jsonl \
    -o submissions/final/eval_patient_doctor_encounters_anonymized.jsonl \
    -d submissions/final/eval_decisions.jsonl
python -m clinanon explain -i data/raw/eval_patient_doctor_encounters.jsonl \
    -d submissions/final/eval_decisions.jsonl -o submissions/final/eval_explained.html
```

- Default profile: rules + spaCy + distilled student, taxonomy `configs/privacy_taxonomy.yaml`.
- Student model `Arsxya/clinanon-student`, revision `0ceeac40131c6cfe9c9b2bbcdbf3d3418c308977`.
- Running it from the local copy or from the Hugging Face download gives byte-identical output.

## Checks

- Same 67 encounter ids in the same order, the same fields, `icd10_codes` byte-identical to the input,
  and the same number of lines in every dialogue and note.
- Local evaluation (`python -m clinanon evaluate`):
  - 100 % of patient names and ages removed (ACI-Bench metadata)
  - 100 % of ICD-description terms kept
  - 99.7 % of clinical-vocabulary tokens kept
- Mid-term comparison: the first version (`submissions/midterm/`) scored privacy 0.918, ICD-10 F1 delta
  0.0 and medical concept recall 0.9996 in the official evaluation.
