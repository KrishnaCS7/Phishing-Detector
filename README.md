# Phishing / Spam Message Detector (NLP project)

TF-IDF + ML classifiers with LIME explainability and a Streamlit demo.

## Run
```bash
pip install -r requirements.txt
python train.py                          # trains on data/sms.tsv, saves model/
streamlit run app.py                     # opens the demo in your browser
```

### Use a real phishing EMAIL dataset (recommended for the final demo)
1. Download "Phishing Email Dataset" from Kaggle (`Phishing_Email.csv`, columns `Email Text`, `Email Type`).
2. Put it at `data/emails.csv`
3. `python train.py --data data/emails.csv` then `streamlit run app.py`

`train.py` auto-detects text/label columns. Do this step on your laptop; keep the SMS run as a second result.

## Files
- `nlp_utils.py` – text cleaning + rule-based red-flag extractor
- `train.py` – trains Naive Bayes / Logistic Regression / Linear SVM, evaluates, saves plots + errors
- `app.py` – Streamlit UI (Detect, Model performance, How it works)
- `model/` – saved model, metrics.json, confusion matrix, top features, errors.csv

## Likely viva questions
- **Why TF-IDF over bag-of-words?** Down-weights common words, up-weights discriminative ones.
- **Why bigrams?** Catches phrases like "click here", "verify account" that single words miss.
- **Why not just accuracy?** Class imbalance; precision/recall/F1 show real performance.
- **What is LIME?** Fits a simple local model around one prediction by perturbing the text.
- **Limitations?** Dataset-specific vocabulary, no header/sender analysis, adversarial rewording. Future work: BERT, URL/header features.
