"""Train + evaluate phishing/spam text classifiers.

Usage:
    python train.py                         # uses data/sms.tsv (demo dataset)
    python train.py --data data/emails.csv  # your own CSV (auto-detects columns)
"""
import argparse, json, os, time
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
from nlp_utils import clean_text

POS = {"spam", "phishing", "phishing email", "malicious", "1", "1.0", "fraud", "scam", "bad", "phish"}
TEXT_CANDS = ["email text", "text_combined", "text", "body", "message", "email_text", "email",
              "content", "v2", "sms", "mail"]
LABEL_CANDS = ["email type", "label", "class", "type", "category", "v1", "target", "spam"]


def load_data(path):
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    else:
        df = pd.read_csv(path, encoding="latin-1")
        cols = {c.lower().strip(): c for c in df.columns}
        print("Columns found:", list(df.columns))
        t = next((cols[c] for c in TEXT_CANDS if c in cols), None)
        l = next((cols[c] for c in LABEL_CANDS if c in cols), None)
        if t is None:  # fall back to the column with the longest average text
            obj = [c for c in df.columns if df[c].dtype == object]
            t = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
        if l is None:
            raise SystemExit(f"Could not find a label column. Columns are: {list(df.columns)}")
        print(f"Using text column = '{t}', label column = '{l}'")
        df = df[[l, t]].rename(columns={l: "label", t: "text"})
    df = df.dropna().drop_duplicates(subset="text")
    df["y"] = df["label"].astype(str).str.lower().str.strip().isin(POS).astype(int)
    df["text"] = df["text"].astype(str)
    print("Label values:", df["label"].astype(str).value_counts().to_dict())
    return df.reset_index(drop=True)


def make_models():
    vec = lambda: TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True, stop_words="english")
    return {
        "Naive Bayes": Pipeline([("tfidf", vec()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", vec()),
            ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))]),
        "Linear SVM": Pipeline([("tfidf", vec()),
            ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sms.tsv")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_data(a.data)
    print(f"Loaded {len(df)} rows | positive (phishing/spam) = {df.y.mean():.1%}")
    X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.y, test_size=0.2,
                                              stratify=df.y, random_state=42)
    results, fitted = {}, {}
    for name, pipe in make_models().items():
        t0 = time.time(); pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te); prob = pipe.predict_proba(X_te)[:, 1]
        results[name] = dict(
            accuracy=accuracy_score(y_te, p), precision=precision_score(y_te, p),
            recall=recall_score(y_te, p), f1=f1_score(y_te, p),
            roc_auc=roc_auc_score(y_te, prob), train_sec=round(time.time() - t0, 2))
        fitted[name] = pipe
        print(f"{name:20s} " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    best = max(results, key=lambda n: results[n]["f1"])
    # Final model = best by F1 (LR preferred on ties: it is the most explainable)
    if abs(results["Logistic Regression"]["f1"] - results[best]["f1"]) < 0.005:
        best = "Logistic Regression"
    print("Selected model:", best)
    joblib.dump(fitted[best], f"{a.out}/model.joblib")

    # confusion matrix
    p = fitted[best].predict(X_te)
    cm = confusion_matrix(y_te, p)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing/Spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best}"); fig.tight_layout()
    fig.savefig(f"{a.out}/confusion_matrix.png", dpi=150); plt.close(fig)

    # top features (only for linear models)
    clf = fitted[best].named_steps["clf"]
    if hasattr(clf, "coef_"):
        names = np.array(fitted[best].named_steps["tfidf"].get_feature_names_out())
        coef = clf.coef_[0]; idx = np.argsort(coef)
        top_neg, top_pos = idx[:15], idx[-15:]
        fig, ax = plt.subplots(figsize=(7, 5))
        sel = np.concatenate([top_neg, top_pos])
        ax.barh(names[sel], coef[sel], color=["#2a9d8f"] * 15 + ["#e63946"] * 15)
        ax.set_title("Most influential words (red = phishing/spam, green = legit)")
        fig.tight_layout(); fig.savefig(f"{a.out}/top_features.png", dpi=150); plt.close(fig)

    # error analysis
    te = pd.DataFrame({"text": X_te.values, "true": y_te.values, "pred": p})
    errs = te[te.true != te.pred]
    errs.to_csv(f"{a.out}/errors.csv", index=False)
    print(f"Misclassified: {len(errs)} / {len(te)}  (FP={((errs.pred==1)).sum()}, FN={((errs.pred==0)).sum()})")

    json.dump({"selected": best, "n_train": len(X_tr), "n_test": len(X_te),
               "dataset": os.path.basename(a.data), "positive_rate": float(df.y.mean()),
               "results": results, "confusion_matrix": cm.tolist()},
              open(f"{a.out}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()