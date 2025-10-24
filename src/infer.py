import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .utils import ID2LABEL

def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    enc = tok(args.texts, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(**enc)
        pred = out.logits.argmax(dim=-1).tolist()
    print([ID2LABEL[p] for p in pred])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="artifacts/checkpoints")
    ap.add_argument("texts", nargs="+", help="One or more sentences to classify")
    args = ap.parse_args()
    main(args)
