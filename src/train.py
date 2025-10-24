import argparse
from datasets import DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, 
                          Trainer, DataCollatorWithPadding, EarlyStoppingCallback)
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

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    use_tb = ["tensorboard"] if args.tb else ["none"]
    eval_strategy = "steps" if args.eval_steps > 0 else "epoch"
    args_tr = TrainingArguments(
        output_dir="artifacts/checkpoints",
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=use_tb,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", tokenized.get("test")),
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()

    print("Evaluating on test/validation...")
    metrics = trainer.evaluate(tokenized.get("test", tokenized.get("validation")))
    print(metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200, help="Use 0 to switch to epoch eval")
    ap.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    args = ap.parse_args()
    main(args)
