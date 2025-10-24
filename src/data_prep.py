import argparse
from datasets import DatasetDict, Dataset
import pandas as pd
from collections import Counter
from .utils import load_financial_phrasebank, detect_text_col, set_seed

def drop_duplicates_ds(ds, text_col):
    df = ds.to_pandas().drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return Dataset.from_pandas(df, preserve_index=False)

def subset(ds: DatasetDict, n_train=3000, n_val=500, n_test=500):
    out = {}
    for k in ds.keys():
        n = {"train": n_train, "validation": n_val, "test": n_test}.get(k, 500)
        n = min(n, len(ds[k]))
        out[k] = ds[k].select(range(n))
    return DatasetDict(out)

def main(args):
    set_seed()
    ds = load_financial_phrasebank()
    text_col = detect_text_col(ds)
    print("Using text column:", text_col)

    # normalize whitespace
    def clean_text(x):
        x[text_col] = " ".join(str(x[text_col]).split())
        return x
    ds = DatasetDict({k: v.map(clean_text) for k, v in ds.items()})
    ds = DatasetDict({k: drop_duplicates_ds(v, text_col) for k, v in ds.items()})

    if args.subset:
        ds = subset(ds, args.n_train, args.n_val, args.n_test)

    # stats
    for k in ds.keys():
        cnt = Counter(ds[k]["label"])
        print(f"{k}: size={len(ds[k])} labels={dict(cnt)}")

    print("Data ready in memory via datasets library. Proceed to training.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    args = ap.parse_args()
    main(args)
