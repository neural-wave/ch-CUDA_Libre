"""Fine-tune the student: a small token classifier for our privacy categories (runs on CPU at inference).

It starts from a clinical de-identification model (StanfordAIMI/stanford-deidentifier-base: BERT-base,
110M parameters, uncased, so lower-cased ASR and cased notes look the same) and learns our taxonomy from
the teacher's data: real encounters labelled by the teacher and encounters the teacher wrote. Texts are
cut into line/sentence-aligned chunks that fit 512 tokens; labels are BIO tags derived from the character
spans. 5% of the documents are held back to report span F1. Trained on one GPU in a few minutes:

    python scripts/train_student.py --data data/distill/train_real_heldout.jsonl \\
        data/distill/teacher_generated_s2.jsonl data/distill/teacher_generated_s3.jsonl \\
        --out models/student --batch 8 --grad-accum 2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

CATEGORIES = ["NAME", "AGE", "DATE", "TIME", "LOCATION", "ORGANIZATION", "PROFESSION", "CONTACT", "IDENTIFIER"]
LABELS = ["O"] + [f"{p}-{c}" for c in CATEGORIES for p in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def text_chunks(text: str, max_words: int = 180):
    """(offset, chunk) pieces split at line breaks and sentence ends (same idea as evaluation.judges.chunks)."""
    import re

    pieces = [m for m in re.finditer(r"[^\n]*(?:\n|$)", text) if m.group()]
    current, start = [], 0
    for m in pieces:
        words = len(m.group().split())
        if current and sum(len(p.group().split()) for p in current) + words > max_words:
            yield start, text[start : current[-1].end()]
            current = []
        if not current:
            start = m.start()
        current.append(m)
        if words > max_words:  # one very long line: split it at sentence ends
            yield from _split_long(text, m.start(), m.end(), max_words)
            current = []
    if current:
        yield start, text[start : current[-1].end()]


def _split_long(text: str, a: int, b: int, max_words: int):
    import re

    start, count = a, 0
    for m in re.finditer(r"[^.!?]*[.!?]+\s*|[^.!?]+$", text[a:b]):
        n = len(m.group().split())
        if count and count + n > max_words:
            yield start, text[start : a + m.start()]
            start, count = a + m.start(), 0
        count += n
    if start < b:
        yield start, text[start:b]


def examples(records: list[dict], tokenizer, max_length: int = 512) -> list[dict]:
    out = []
    for r in records:
        for field in ("dialogue", "note"):
            text = r[field]
            spans = sorted((s for s in r["spans"] if s["field"] == field), key=lambda s: s["start"])
            for offset, chunk in text_chunks(text):
                if not chunk.strip():
                    continue
                enc = tokenizer(chunk, truncation=True, max_length=max_length, return_offsets_mapping=True)
                labels = []
                for a, b in enc["offset_mapping"]:
                    if a == b:
                        labels.append(-100)  # special tokens
                        continue
                    label = "O"
                    for s in spans:
                        if s["start"] <= offset + a < s["end"]:
                            begin = offset + a == s["start"] or not text[s["start"] : offset + a].strip()
                            label = f"{'B' if begin else 'I'}-{s['category']}"
                            break
                    labels.append(LABEL2ID[label])
                enc.pop("offset_mapping")
                enc["labels"] = labels
                out.append(dict(enc))
    return out


def span_f1(predictions, labels) -> dict:
    """Exact-match span precision/recall/F1 from BIO tags (per chunk)."""

    def spans(seq):
        found, current = set(), None
        for i, tag in enumerate(seq):
            if tag.startswith("B-") or (tag.startswith("I-") and (current is None or current[2] != tag[2:])):
                if current:
                    found.add(tuple(current))
                current = [i, i + 1, tag[2:]]
            elif tag.startswith("I-") and current:
                current[1] = i + 1
            else:
                if current:
                    found.add(tuple(current))
                current = None
        if current:
            found.add(tuple(current))
        return found

    tp = fp = fn = 0
    per_cat = Counter()
    for pred, gold in zip(predictions, labels, strict=True):
        keep = [i for i, g in enumerate(gold) if g != -100]
        p = spans([LABELS[pred[i]] for i in keep])
        g = spans([LABELS[gold[i]] for i in keep])
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
        for s in g:
            per_cat[(s[2], s in p)] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"precision": precision, "recall": recall, "f1": f1}
    for c in CATEGORIES:
        n = per_cat[(c, True)] + per_cat[(c, False)]
        if n:
            out[f"recall_{c}"] = per_cat[(c, True)] / n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--base", default="StanfordAIMI/stanford-deidentifier-base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    records = [json.loads(line) for path in args.data for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    random.shuffle(records)
    n_dev = max(10, len(records) // 20)
    dev_records, train_records = records[:n_dev], records[n_dev:]
    print(f"{len(train_records)} train / {len(dev_records)} dev documents from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    train, dev = examples(train_records, tokenizer), examples(dev_records, tokenizer)
    print(f"{len(train)} train / {len(dev)} dev chunks")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        ignore_mismatched_sizes=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return span_f1(np.argmax(logits, axis=-1).tolist(), labels.tolist())

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=16,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=int(0.06 * args.epochs * len(train) / (args.batch * args.grad_accum)),
            weight_decay=0.01, lr_scheduler_type="linear", eval_strategy="epoch", save_strategy="no",
            logging_steps=50, bf16=torch.cuda.is_available(), report_to=[], seed=args.seed,
        ),
        train_dataset=train, eval_dataset=dev, data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "dev_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
