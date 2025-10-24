import argparse, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from datasets import DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, 
                          Trainer, DataCollatorWithPadding)
from .utils import (MODEL_NAME, ID2LABEL, LABEL2ID, load_financial_phrasebank,
                    detect_text_col, compute_metrics, set_seed)

def coerce_labels(ds):
    def fix(example):
        lbl = example["label"]
        if isinstance(lbl, str):
            m = {"negative":0, "neutral":1, "positive":2}
            lbl = m.get(lbl.lower(), 1)
        example["label"] = int(lbl)
        return example
    return ds.map(fix)

def get_tokenized(ds, text_col):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def tokenize_fn(batch): return tok(batch[text_col], truncation=True)
    tokenized = DatasetDict()
    for split in ds.keys():
        rem = [c for c in ds[split].column_names if c == text_col]
        tokenized[split] = ds[split].map(tokenize_fn, batched=True, remove_columns=rem)
    for split in tokenized.keys():
        tokenized[split] = coerce_labels(tokenized[split])
    collator = DataCollatorWithPadding(tokenizer=tok)
    return tok, tokenized, collator

def evaluate_model(model_name_or_path, tokenized, tok, collator):
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
        ),
        args=TrainingArguments(output_dir="artifacts/metrics", report_to=["none"]),
        eval_dataset=tokenized.get("test", tokenized.get("validation")),
        tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
    )
    return trainer.evaluate()

def main(args):
    set_seed()
    ds = load_financial_phrasebank()
    text_col = detect_text_col(ds)
    tok, tokenized, collator = get_tokenized(ds, text_col)

    if args.baseline:
        print("Evaluating baseline (no fine-tune)")
        print(evaluate_model(MODEL_NAME, tokenized, tok, collator))

    print("Evaluating fine-tuned (if artifacts exist, else this reuses base)")
    print(evaluate_model("artifacts/checkpoints", tokenized, tok, collator))

    # Error analysis
    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained("artifacts/checkpoints", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID),
        args=TrainingArguments(output_dir="artifacts/metrics", report_to=["none"]),
        eval_dataset=tokenized.get("test", tokenized.get("validation")),
        tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
    )
    preds = trainer.predict(tokenized.get("test", tokenized.get("validation")))
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)

    mistakes = np.where(y_true != y_pred)[0][:10]
    rows = []
    orig_name = "test" if "test" in ds else "validation"
    for i in mistakes:
        ex = ds[orig_name][int(i)]
        field = "sentence" if "sentence" in ex else ("text" if "text" in ex else None)
        rows.append({"text": ex.get(field, ""), "true": ID2LABEL[int(y_true[i])], "pred": ID2LABEL[int(y_pred[i])]})
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    main(args)
