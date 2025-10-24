import gradio as gr, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils import ID2LABEL

MODEL_DIR = "artifacts/checkpoints"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def infer_one(text):
    enc = tok([text], return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(**enc)
        pred = out.logits.softmax(dim=-1).squeeze().tolist()
    return {ID2LABEL[i]: float(p) for i, p in enumerate(pred)}

gr.Interface(fn=infer_one, inputs="text", outputs=gr.Label(num_top_classes=3),
             title="Financial Sentiment (DistilBERT)").launch()
