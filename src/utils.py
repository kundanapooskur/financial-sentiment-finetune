import numpy as np, torch, random
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"
LABELS = {0:"negative", 1:"neutral", 2:"positive"}
ID2LABEL = {0:"negative", 1:"neutral", 2:"positive"}
LABEL2ID = {"negative":0, "neutral":1, "positive":2}

def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_financial_phrasebank():
    """Prefer a Parquet-backed dataset (no deprecated scripts). Provides train/validation/test."""
    try:
        ds = load_dataset("atrost/financial_phrasebank")
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.1, seed=RANDOM_SEED)
            ds = DatasetDict({"train": split["train"], "validation": split["test"], "test": ds.get("test", split["test"])})
        return ds
    except Exception as e:
        print("Fallback to takala Parquet due to:", e)
        ds = load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/takala/financial_phrasebank/sentences_allagree/financial_phrasebank-train.parquet",
                "test":  "hf://datasets/takala/financial_phrasebank/sentences_allagree/financial_phrasebank-test.parquet"
            }
        )
        split = ds["train"].train_test_split(test_size=0.1, seed=RANDOM_SEED)
        return DatasetDict({"train": split["train"], "validation": split["test"], "test": ds["test"]})

def detect_text_col(ds):
    sample = ds["train"][0]
    for cand in ["sentence", "text", "content"]:
        if cand in sample:
            return cand
    raise ValueError(f"No text column found; keys: {list(sample.keys())}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {"accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}
