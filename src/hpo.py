import argparse, itertools, pandas as pd
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

def main(args):
    set_seed()
    ds = load_financial_phrasebank()
    text_col = detect_text_col(ds)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(batch): return tok(batch[text_col], truncation=True)
    tokenized = DatasetDict()
    for split in ds.keys():
        rem = [c for c in ds[split].column_names if c == text_col]
        tokenized[split] = ds[split].map(tokenize_fn, batched=True, remove_columns=rem)
    for split in tokenized.keys():
        tokenized[split] = coerce_labels(tokenized[split])
    collator = DataCollatorWithPadding(tokenizer=tok)

    grid = {
        "learning_rate": [5e-6, 1e-5, 2e-5],
        "num_train_epochs": [2, 3],
        "per_device_train_batch_size": [8, 16]
    }
    results = []
    for lr, epochs, bs in itertools.product(grid["learning_rate"], grid["num_train_epochs"], grid["per_device_train_batch_size"]):
        out_dir = f"artifacts/hpo_lr{lr}_ep{epochs}_bs{bs}"
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
        args_tr = TrainingArguments(
            output_dir=out_dir, evaluation_strategy="epoch", save_strategy="no",
            learning_rate=lr, num_train_epochs=epochs, per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
            weight_decay=0.01, logging_steps=100, report_to=["none"], seed=42
        )
        trainer = Trainer(
            model=model, args=args_tr, train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation", tokenized.get("test")),
            tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
        )
        trainer.train()
        m = trainer.evaluate(tokenized.get("test", tokenized.get("validation")))
        results.append({"lr": lr, "epochs": epochs, "batch": bs,
                        "eval_accuracy": m.get("eval_accuracy"),
                        "eval_f1_macro": m.get("eval_f1_macro"),
                        "eval_loss": m.get("eval_loss")})

    df = pd.DataFrame(results).sort_values(["eval_f1_macro","eval_accuracy"], ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="Run simple grid search")
    args = ap.parse_args()
    main(args)
