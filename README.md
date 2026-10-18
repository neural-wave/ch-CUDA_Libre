# clinanon: ICD-aware anonymization of patient–doctor encounters

Bosch challenge @ Neural Wave Hackathon 2026. `clinanon` removes privacy-related information from
doctor–patient dialogues and clinical notes while keeping the ICD-10-relevant medical content intact.
It runs fully on-prem on a CPU (no GPU, about 0.6 s per encounter), and every decision comes with a
written explanation.

**Team:** Arslan Ali, Daniel Sima, Stipe Peran, Kristina Marinova

**Deliverables:** the report and the video link are in [`deliverables/`](deliverables/).

## Results at a glance

| | |
|---|---|
| Privacy, eval set | 100 % of patient names and ages removed (ACI-Bench metadata) |
| Medical information, eval set | ICD-10 coding F1 0.193 on the original, 0.195 on the anonymized text (LLM coder); 99.7 % of medical concepts and 100 % of ICD-description terms kept; ICD codes untouched |
| Synthetic encounters with inserted names, IDs, phones, addresses (test split) | **94.4 %** of private details removed |
| Unseen document types: radiology, pathology, ICU, operative reports, portal messages (test set) | **89.0 %** of private details removed |
| Runtime | 0.59 s per encounter on a laptop CPU, 0 GB VRAM; 0.02 s in the rules-only profile |

The submitted file is
[`submissions/final/eval_patient_doctor_encounters_anonymized.jsonl`](submissions/final/). Its
decisions and explanations sit next to it, with an HTML view and a manifest that holds checksums and
the exact command.

## Quick start

Python 3.11 or 3.12.

```bash
pip install -r requirements.txt     # or: uv sync
# CPU-only torch (smaller download): add --extra-index-url https://download.pytorch.org/whl/cpu

# Anonymize (output keeps the input schema; icd10_codes are never modified)
python -m clinanon anonymize -i data/raw/eval_patient_doctor_encounters.jsonl \
    -o outputs/eval_anonymized.jsonl -d outputs/eval_decisions.jsonl

# Evaluate against the original (privacy, utility proxies, damage audit)
python -m clinanon evaluate -i data/raw/eval_patient_doctor_encounters.jsonl -a outputs/eval_anonymized.jsonl

# See every decision with its explanation (HTML, hover a span)
python -m clinanon explain -i data/raw/eval_patient_doctor_encounters.jsonl \
    -d outputs/eval_decisions.jsonl -o outputs/eval_explained.html
```

- **Student model download:** on first run the distilled student model (436 MB) is downloaded from
  Hugging Face ([`Arsxya/clinanon-student`](https://huggingface.co/Arsxya/clinanon-student), pinned
  revision) into the local cache. A copy in `models/student/` is used instead if present. Without
  network the pipeline runs rules + spaCy only and logs a warning.
- **Options:**
  - `--workers N`: parallel processing.
  - `--no-student` / `--no-spacy`: faster profiles (see Efficiency).
  - `--no-icd`: do not read the ICD codes.
  - `--keep-asr-splits`: leave "wan na" as it is.

## How it works

```
record ─► rules + spaCy NER + distilled student ─► clinical veto ─► names and places propagated
       ─► clinical veto (context) ─► overlap resolution ─► typed tags ─► record' + decisions with explanations
```

| Stage | What it does |
|---|---|
| **Rules** (`detectors/rules.py`) | Covers names in context (titles, greetings, dictation headers, "X is a 59-year-old"), ages, dates and times, IDs, phones and e-mails (also spoken), addresses and postcodes, named facilities, employers and occupations, and hospital form fields ("MRN:", "SURGEON:"). Cities, regions and countries come from GeoNames and match only after "in", "from", "to" or a postcode. Clinical look-alikes never match: "2/6" murmurs, "7/10" pain, "120/80", the verb "may". |
| **spaCy NER** (`detectors/ner.py`) | spaCy via Presidio for names, places and organizations. Lower-cased ASR dialogues are first **truecased with the help of the encounter's note**, keeping their length so offsets map 1:1. |
| **Distilled student** (`detectors/student.py`) | A 110M-parameter token classifier on CPU, trained on data labelled and written by a 27B open LLM. It finds names, places, organizations and jobs from context, including European names in lower-cased speech. |
| **Clinical veto** (`veto.py`) | Drops weak candidates that are clinical: eponyms in context ("Parkinson's disease", "positive Spurling's"), drugs (FDA list, or any name followed by a dose), ICD-10-CM vocabulary, and abbreviations. |
| **Propagation** (`propagation.py`) | A name or facility found once (often in the cased note) is removed everywhere in the encounter, including spoken short forms ("east metro"). A name that is also a clinical term only propagates where a title marks it. |
| **Policy** (`configs/privacy_taxonomy.yaml`) | Replaces each span with a typed tag (`[PATIENT]`, `[DOCTOR]`, `[AGE]`, `[DATE]`, …). |
| **Explanations** (`explain.py`) | Every decision gets one sentence: what was found, by which evidence, why it is private (with its literature source), and what was done. |

### Design decisions

- **Tags, not generalized values or fake names.** Privacy is scored by a PII detector run over the
  output. Detectors ignore `[DATE]`, but they flag "1958", "in his 50s" or a realistic fake name.
- **Minimal diff.** Only detected spans change. There is no free-text rewriting, because LLM rewriters
  alter clinical facts. Speaker tags and section headers are protected.
- **What counts as private** is documented in
  [`configs/privacy_taxonomy.yaml`](configs/privacy_taxonomy.yaml), with literature sources (HIPAA
  Safe Harbor, i2b2 2014, GDPR, TAB, the Stanford de-identification labels). The same file configures
  the policy.
  - **Tagged:** names; ages; every calendar element (dates, months, weekdays, holidays, seasons,
    spoken years); clock times; addresses and named places; named facilities, employers and insurers;
    occupations; contact details; identifiers; the names of documentation assistants ("hey dragon").
  - **Kept on purpose:**
    - relative time ("today", "two weeks ago") and frequencies
    - clock-face positions in an exam
    - types of place ("the emergency room", "the beach"), which external-cause codes need
    - care-team roles and relatives ("my cardiologist", "my dad")
    - occupations that are part of a diagnosis ("farmer", as in farmer's lung)
- **ASR contraction repair.** "wan na", "lem me" and "gon na" become "wanna", "lemme" and "gonna".
  Detectors read the first half as a person's name; rejoining changes no information.

## Distillation: a large open model teaches a small one

Inference must run on-prem with at most 4 GB of VRAM; development need not. So a large open model is
the teacher at development time, and the small model it trains, the student, is what ships.

- **Teacher:** Qwen3.8-27B (Apache 2.0), served with vLLM on two rented GPUs.
  - It labels the 140 held-out ACI encounters by copying each chunk with inline tags. A chunk counts
    only if removing the tags gives back the input exactly (1,274 of 1,291 chunks). Our rule-based
    ages and dates are added, so the labels follow our taxonomy.
  - It writes 1,066 new encounters: 25 settings, 18 patient origins, 6 note formats, and clinical
    traps (eponyms, drugs, relative time, injury places). A separate labelling pass labels them.
  - A filter built on our clinical veto drops eponyms, drugs and ICD vocabulary from the labels.
- **Student:** `StanfordAIMI/stanford-deidentifier-base` (BERT-base, uncased, already trained for
  clinical de-identification), fine-tuned in minutes on one GPU. It agrees with the teacher at span
  F1 0.974 on held-back documents.
- **What the student is trusted with:** only names, places, organizations and jobs, the categories
  that need context. Ages, dates, IDs and contacts are formats the rules cover precisely; the student
  read lab values as ages. Its spans cover whole words, need a confidence of 0.75 or more, and still
  pass through the clinical veto.

| | Rules + spaCy | + GLiNER (zero-shot) | **+ distilled student** |
|---|---|---|---|
| Synthetic test, details removed | 90.4 % | 91.6 % | **94.4 %** |
| Real eval set: names/ages removed, ICD terms kept, clinical words kept | 100 / 100 / 99.7 % | removes pronouns, anatomy, injury places | 100 / 100 / 99.7 % |
| CPU time per encounter | 0.25 s | 1.9 s | 0.59 s |

On the 67 real eval encounters, which were never used for training, the student adds 7 tags such as
"northern cascades" and "landscapes warehouse". On the synthetic test it raises jobs from 30 to 70 %
and employers from 72 to 84 %. Scripts:
- `scripts/teacher_data.py`
- `scripts/merge_rule_spans.py`
- `scripts/train_student.py`

## Evaluation

`clinanon evaluate` reports:

- **Gold privacy:** removal of the patient's first name, last name and age, taken from the ACI-Bench
  metadata, independently of our detectors.
- **Judge privacy:** the share of values found by stand-in PII detectors (Presidio) that are gone from
  the output.
- **Utility proxies:** how many ICD-description terms and clinical-vocabulary tokens are still present.
- **Damage audit:** every edit, with suspected clinical removals flagged.

`scripts/calibrate_judges.py` scores an output under 10 stand-in judges: Presidio, GLiNER PII models,
and clinical de-identification models. We used it to understand the mid-term score; no off-the-shelf
judge reproduces it. On the real data (rules + spaCy configuration):

| Split | Gold names/ages | Presidio-strict | De-id i2b2 (first version → now) | De-id Stanford (first version → now) | Clinical tokens kept |
|---|---|---|---|---|---|
| eval (67, used for tuning) | 100 % | 84.9 % | 75.5 → 97.1 % | 60.0 → 84.7 % | 99.7 % |
| other (140, held out) | 100 % | 86.8 % | 76.2 → 92.2 % | 66.6 → 86.1 % | 99.8 % |

What still survives the judges is mostly vocabulary we keep on purpose: clinical terms ("Cushing's",
"Lasix"), interjections the models misread, relatives and care-team roles, and types of place.

**Synthetic encounters with known answers.** `scripts/make_synthetic.py` builds:
- two splits (dev and test) of 40 held-out ACI encounters each;
- in each, the patient gets a European, British or US name;
- each gets inserted exchanges and note passages with administrative details: date of birth, phone,
  e-mail, address, insurer, AHV and insurance numbers, employer, relatives, other doctors, appointments.

A local LLM wrote the wording with placeholders, and Faker and real place names filled them, so every
value (3,181 in total) and its offsets are known. Half of the dialogues are lower-cased ASR; some
details are spoken. We inspected misses on dev only. `scripts/eval_synthetic.py` scores a run. The test
split went from 71.1 % at the start of this work to 94.4 % now.

## Bonus tasks

### Adversarial privacy attacks (`scripts/attack.py`, the 27B model as attacker)

- **Recovery:** every tag gets a number, and the attacker guesses the original value from context.
  - Overall it recovers 6.1 % of 776 removed values exactly and 11.6 % approximately.
  - Names: 0.3 %. Ages: 1.8 % exactly, 24 % within 3 years.
  - Most exact "recoveries" are words that were never private but were tagged by mistake ("Medicare").
- **Linking:** given an anonymized encounter and 10 candidate patients (name, age, sex), the attacker
  picks the right one in 22.4 % of cases. Chance is 10 %; sex is kept on purpose, which alone gives
  about 20 %.
- **Linking with background knowledge:** from each original encounter the teacher writes what an
  acquaintance could know, non-medical facts only (age, job, hobbies, trips, events). The attacker then
  ranks all 67 anonymized encounters for each profile.

  | Text | Correct encounter ranked first | Mean reciprocal rank |
  |---|---|---|
  | Original | 89.2 % | 0.93 |
  | Mid-term output | 75.6 % | 0.81 |
  | Final output | 73.6 % | 0.80 |

  This is the limit of span-level de-identification. The attacker links on the story ("moving boxes on
  the basement stairs", "a minor car accident at a four-way stop"). None of these details is an
  identifier under HIPAA or i2b2, and some, like how an injury happened, are what external-cause ICD-10
  codes need. Among 67 encounters almost every story is unique; in a hospital-sized population a
  profile matches many patients. Reducing this risk means generalizing the narrative, which costs
  utility.

### Explainability

- **Coverage:** every decision carries an `explanation`, so 100 % of the 1,377 decisions on the eval
  set are explained. This includes rejected candidates. Example:

  > "Parkinson" is a person's name (direct identifier; HIPAA, I2B2, TAB); evidence: it is the subject
  > of a sentence stating an age; replaced by [PATIENT].

  Its rejected twin:

  > Candidate rejected: "Parkinson" was proposed as name (evidence: the same name or place was
  > identified elsewhere in this encounter), but it is followed by a clinical head noun (disease, sign,
  > test, joint ...), so it names a clinical concept.
- **Viewer:** `clinanon explain` writes an HTML page that highlights every removed and kept span, with
  its explanation on hover.
- **Correctness:** `scripts/check_explanations.py` has the 27B model audit a stratified sample of 120
  decisions. Weighted by category frequency:
  - removals: decisions 93.6 % correct, explanations 90.8 %
  - kept spans: decisions 100 % correct, explanations 91.4 %

  Most objections concern labels, not leaks.

### Efficiency (`scripts/benchmark.py`, laptop CPU, 0 GPU)

| Profile | Per encounter | Peak RAM | Gold names/ages | Synthetic test | Clinical words kept |
|---|---|---|---|---|---|
| fast: rules only (`--no-spacy --no-student`) | **0.02 s** | 0.4 GB | 100 % | 83.2 % | 99.8 % |
| balanced: rules + student (`--no-spacy`) | 0.39 s | 2.1 GB | 100 % | 93.8 % | 99.7 % |
| rules + spaCy (`--no-student`) | 0.25 s | 1.4 GB | 100 % | 90.4 % | 99.7 % |
| **default**: rules + spaCy + student | 0.59 s | 2.8 GB | 100 % | **94.4 %** | 99.7 % |

- `--workers N` gives the same output in parallel. With 4 workers: 0.43 s per encounter, model loading
  included.
- int8 quantization of the student was tried: it was 2.7× slower on Apple silicon and changed outputs,
  so we dropped it.

### Without ICD-10 codes at inference

`--no-icd` never reads the codes. On all 207 ACI encounters the output is **byte-identical** with and
without them, because the global ICD-10-CM vocabulary and eponym list already protect every clinical
word.

With the 27B model as an ICD coder (`scripts/icd_check.py`), anonymization costs no coding accuracy:

| Text | Coding F1 |
|---|---|
| Original | 0.193 |
| Mid-term output | 0.195 |
| Final output | 0.195 |

The 27B model is weak at exact codes, but the result is consistent with the official F1 delta of 0.0.

### Robustness to other input data

`scripts/teacher_data.py robust` writes held-out documents of types the student never saw:
- documents: radiology, pathology, operative and physiotherapy reports; ICU nursing notes; psychiatric
  admissions; home-care reports; fitness-for-work assessments; portal messages; medication
  reconciliations
- paired conversations: phone triage, handovers, ward rounds, pharmacist counselling, paramedic
  handovers, and interpreter-mediated visits

Each comes with labels and ICD-10 codes from the teacher (`data/robustness/`). The misses on the dev
set led to general rules for labelled form fields and "14-MAR-1949" dates. The test set was generated
before those rules and scored once:

| Robustness set | Rules + spaCy | Final |
|---|---|---|
| dev (119 documents, misses inspected) | 82.3 % | 89.9 % |
| **test** (116 documents, untouched) | 86.1 % | **89.0 %** |

On the test set, 99.0 % of ICD-description terms are kept. The weakest categories are IDs spoken in
conversations and organizations. The labels come from the teacher, so they are silver, not gold.

## Repository layout

```
configs/privacy_taxonomy.yaml   privacy categories (with sources) + replacement policy
src/clinanon/
  cli.py                        `clinanon anonymize | evaluate | explain`
  pipeline.py                   orchestration, overlap resolution, replacement
  detectors/                    rules.py, ner.py (spaCy), student.py (distilled), gliner_ner.py (optional)
  veto.py, propagation.py       clinical veto, record-level propagation
  explain.py                    explanations and the HTML viewer
  distill/teacher.py            teacher prompts, tag parsing, OpenAI-compatible client
  resources/                    name lists, FDA drugs, ICD-10-CM vocabulary, GeoNames places, curated terms
  evaluation/                   gold (ACI metadata), stand-in judges, metrics, damage audit
scripts/                        build_resources, make_synthetic / eval_synthetic, teacher_data,
                                merge_rule_spans, train_student, attack, icd_check,
                                check_explanations, benchmark, calibrate_judges, ablation
data/raw/                       challenge data (ACI-Bench encounters with ICD-10 codes)
data/external/                  ACI-Bench metadata (gold names/ages for evaluation)
data/synthetic/                 synthetic test encounters with gold values
data/distill/                   student training data (teacher labels and teacher-written encounters)
data/robustness/                held-out documents in unseen formats (dev, test)
submissions/                    final/ (the anonymized eval set) and midterm/
deliverables/                   report (PDF) and video link
tests/                          regression tests (pytest)
```

## Data, models and tools

- **Encounters:** ACI-Bench (Yim et al., 2023; CC BY 4.0), with ICD-10-CM annotations from Yuan et al.,
  EMNLP 2025 Industry Track (CC BY-NC 4.0), as provided by the organizers.
- **Lexicons:**
  - US Census 1990/2010 name lists, SSA baby names, FDA Drugs@FDA and CDC/NCHS ICD-10-CM 2025 (US
    government works)
  - GeoNames cities, countries and regions (CC BY 4.0)

  Rebuild them with `python scripts/build_resources.py`.
- **Models:**
  - spaCy `en_core_web_lg` (MIT)
  - Microsoft Presidio (MIT)
  - StanfordAIMI/stanford-deidentifier-base (MIT), the base of our student
  - Qwen3.8-27B (Apache 2.0), the teacher, attacker and auditor
  - Qwen3.5-9B, which wrote the synthetic test passages
  - obi/deid_roberta_i2b2 and GLiNER PII models (nvidia, urchade, knowledgator), as stand-in judges
  - our student: [`Arsxya/clinanon-student`](https://huggingface.co/Arsxya/clinanon-student)
    (CC BY-NC 4.0, because of its training data)
- **Tools:** vLLM (serving the teacher), Hugging Face transformers, Faker (synthetic values).
- **Synthetic and teacher-written data:** all persons, numbers and places in it are fictitious.
- **License:** the code is released under the MIT license (`LICENSE`); data files keep the licenses
  above.

## Report and literature

The 3-page report is [`deliverables/clinanon_report.pdf`](deliverables/clinanon_report.pdf); the
2-minute video is linked in [`deliverables/README.md`](deliverables/README.md).

Literature we build on:

- **Data**
  - W. Yim et al. ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic
    Visit Note Generation. *Scientific Data* 10:586, 2023.
  - Yuan, Moy et al. Toward Reliable Clinical Coding with Language Models: Verification and Lightweight
    Adaptation. *EMNLP 2025 Industry Track*.
- **What counts as private**
  - U.S. HHS. Guidance Regarding Methods for De-identification of Protected Health Information in
    Accordance with the HIPAA Privacy Rule, 2012.
  - A. Stubbs, Ö. Uzuner. Annotating longitudinal clinical narratives for de-identification: The 2014
    i2b2/UTHealth corpus. *J. Biomed. Inform.* 58, 2015.
  - I. Pilán et al. The Text Anonymization Benchmark (TAB). *Computational Linguistics* 48(4), 2022.
  - P. Lison et al. Anonymisation Models for Text Data: State of the art, Challenges and Future
    Directions. *ACL-IJCNLP 2021*.
  - Regulation (EU) 2016/679 (GDPR).
- **Clinical de-identification**
  - I. Neamatullah et al. Automated de-identification of free-text medical records. *BMC Med. Inform.
    Decis. Mak.* 8:32, 2008.
  - F. Dernoncourt et al. De-identification of patient notes with recurrent neural networks. *JAMIA*
    24(3), 2017.
  - P. J. Chambon et al. Automated deidentification of radiology reports combining transformer and "hide
    in plain sight" rule-based methods. *JAMIA* 30(2), 2023. Our student starts from their model.
  - D. Carrell et al. Hiding in plain sight: use of realistic surrogates to reduce exposure of protected
    health information in clinical text. *JAMIA* 20(2), 2013.
- **Models and methods**
  - G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531,
    2015.
  - W. Zhou et al. UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity
    Recognition. *ICLR 2024*.
  - J. Devlin et al. BERT. *NAACL 2019*.
  - U. Zaratiana et al. GLiNER: Generalist Model for Named Entity Recognition using Bidirectional
    Transformer. *NAACL 2024*.
  - W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention
    (vLLM). *SOSP 2023*.
  - Qwen Team. Qwen3 Technical Report. arXiv:2505.09388, 2025.
  - L. V. Lita et al. tRuEcasIng. *ACL 2003*.
- **Privacy attacks**
  - L. Sweeney. k-anonymity: A model for protecting privacy. *IJUFKS* 10(5), 2002.
  - A. Narayanan, V. Shmatikov. Robust De-anonymization of Large Sparse Datasets. *IEEE S&P 2008*.
  - R. Staab et al. Beyond Memorization: Violating Privacy via Inference with Large Language Models.
    *ICLR 2024*.

## Use of AI agents

Claude Code (Anthropic) was our coding assistant for implementation, experiments and documentation; the team made
and reviewed every design and policy decision. The open-weight LLMs described above (teacher, attacker, auditor)
are used only at development time; none runs when anonymizing.
