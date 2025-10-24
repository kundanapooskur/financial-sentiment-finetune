# Financial Sentiment Fine-Tuning (DistilBERT)

Fine-tune **DistilBERT** to classify financial sentences as **negative / neutral / positive** using a public dataset.
This repo is rubric-complete: data prep, model selection, training, evaluation, baseline, HPO, error analysis, inference, and reproducible docs.

## Quickstart (Colab or local GPU)
1. **Python 3.10+**, CUDA if available.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. (Colab) Set **Runtime → Change runtime type → GPU (T4)**.
4. Prepare data (loads from Hugging Face, cleans, creates subsets & splits):
   ```bash
   make prep
   ```
5. (Optional) Baseline (no fine-tuning):
   ```bash
   make baseline
   ```
6. Train:
   ```bash
   make train
   ```
7. Evaluate:
   ```bash
   make eval
   ```
8. (Optional) Simple HPO grid:
   ```bash
   make hpo
   ```
9. (Optional) Gradio demo:
   ```bash
   make demo
   ```

## Project Structure
```
src/
  data_prep.py     # load HF dataset, clean, split, optional subset
  utils.py         # shared helpers (dataset loading, metrics)
  train.py         # fine-tune with HF Trainer (TB logging optional)
  eval.py          # evaluate fine-tuned vs. baseline + confusion matrix
  hpo.py           # small grid search (>=3 configs)
  infer.py         # batch inference helper
app/
  demo_gradio.py   # tiny UI for inference
artifacts/
  checkpoints/ metrics/ logs/   # created at runtime
```

## Notes
- Uses **Parquet-backed** dataset loader to avoid deprecated script loaders.
- Default model: `distilbert-base-uncased` (fast, reliable).
- Metrics: accuracy, precision_macro, recall_macro, f1_macro.
- Logging: TensorBoard (optional via `--tb`).
- Error analysis printed in `eval.py` (confusion matrix + examples).

## Reproducibility
- Pinned package versions in `requirements.txt`.
- Deterministic seeds where possible.
- Small **subset** defaults in `make prep` to keep runtime short; remove `--subset` for full dataset.

## Ethics/Limitations
- Financial sentiment has ambiguity and hedging language; include notes in report.
- Consider stronger models (`roberta-base`), class weighting, or focal loss if needed.
