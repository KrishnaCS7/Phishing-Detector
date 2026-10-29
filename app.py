import html, json, os, re
import joblib, pandas as pd, streamlit as st
from lime.lime_text import LimeTextExplainer
from nlp_utils import clean_text, extract_indicators  # noqa: F401 (needed to unpickle)

st.set_page_config(page_title="Phishing Detector", page_icon="🎣", layout="wide")


@st.cache_resource
def load():
    model = joblib.load("model/model.joblib")
    metrics = json.load(open("model/metrics.json"))
    return model, metrics


model, metrics = load()
explainer = LimeTextExplainer(class_names=["Legit", "Phishing"], random_state=42)

SAMPLES = {
    "Bank phishing": "URGENT: Your account has been suspended due to unusual activity. Verify your password immediately at http://secure-bank-login.xyz/verify or your account will be permanently locked within 24 hours.",
    "Prize scam": "Congratulations!!! You have been selected as a WINNER of a free iPhone. Click here to claim your prize now: http://bit.ly/3xFr33 Limited time offer!",
    "Normal work email": "Hi team, attaching the minutes from today's meeting. Please review the action items and let me know if I missed anything. We'll sync again on Thursday at 3pm.",
    "Friendly message": "Hey, are we still on for lunch tomorrow? I was thinking of trying that new place near the library. Let me know what time works for you.",
}


def highlight(text, weights):
    """Wrap each word in a coloured <span> according to its LIME weight."""
    out = []
    for tok in re.split(r"(\W+)", text):
        w = weights.get(tok.lower()) if tok.strip() else None
        safe = html.escape(tok)
        if w and abs(w) > 0.01:
            a = min(abs(w) * 4, 0.85)
            c = f"rgba(230,57,70,{a})" if w > 0 else f"rgba(42,157,143,{a})"
            out.append(f"<span style='background:{c};padding:1px 3px;border-radius:3px'>{safe}</span>")
        else:
            out.append(safe)
    return "<div style='line-height:2;font-size:1.05rem'>" + "".join(out).replace("\n", "<br>") + "</div>"


st.title("🎣 Phishing / Spam Message Detector")
st.caption("NLP pipeline: text cleaning → TF-IDF (uni+bi-grams) → classifier → LIME explanation")
tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Model performance", "📘 How it works"])

with tab1:
    st.selectbox("Load a sample", ["(type your own)"] + list(SAMPLES), key="sample")
    default = SAMPLES.get(st.session_state.sample, "")
    text = st.text_area("Paste an email or SMS", value=default, height=170,
                        key=f"txt_{st.session_state.sample}")
    if st.button("Analyze", type="primary") and text.strip():
        prob = float(model.predict_proba([text])[0, 1])
        c1, c2 = st.columns([1, 2])
        with c1:
            if prob >= 0.5:
                st.error(f"⚠️ PHISHING / SPAM\n\nConfidence: {prob:.1%}")
            else:
                st.success(f"✅ LEGITIMATE\n\nConfidence: {1 - prob:.1%}")
            st.progress(prob, text=f"Phishing probability: {prob:.1%}")
        with c2:
            ind = extract_indicators(text)
            st.markdown("**Rule-based red flags**")
            if ind["flags"]:
                for f in ind["flags"]:
                    st.markdown(f"- 🚩 {f}")
            else:
                st.markdown("- None detected")
            if ind["urls"]:
                st.markdown("**Links found:** " + ", ".join(f"`{u}`" for u in ind["urls"]))
        st.subheader("Why did the model decide this?")
        with st.spinner("Running LIME explanation..."):
            exp = explainer.explain_instance(text, model.predict_proba, num_features=12, num_samples=800)
        weights = {w.lower(): v for w, v in exp.as_list()}
        st.markdown(highlight(text, weights), unsafe_allow_html=True)
        st.caption("🔴 pushes towards PHISHING   🟢 pushes towards LEGIT   (darker = stronger)")
        st.dataframe(pd.DataFrame(exp.as_list(), columns=["word", "influence"]).round(3),
                     hide_index=True)

with tab2:
    st.write(f"Dataset: **{metrics['dataset']}** | train = {metrics['n_train']}, test = {metrics['n_test']} "
             f"| phishing/spam share = {metrics['positive_rate']:.1%}")
    df = pd.DataFrame(metrics["results"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4)
    st.dataframe(df)
    st.info(f"Selected model: **{metrics['selected']}**")
    a, b = st.columns(2)
    if os.path.exists("model/confusion_matrix.png"):
        a.image("model/confusion_matrix.png")
    if os.path.exists("model/top_features.png"):
        b.image("model/top_features.png")
    if os.path.exists("model/errors.csv"):
        with st.expander("Error analysis: misclassified test messages"):
            st.dataframe(pd.read_csv("model/errors.csv"))

with tab3:
    st.markdown("""
**Pipeline**
1. **Cleaning** – lowercase, URLs/emails/numbers replaced by placeholder tokens, HTML and punctuation removed.
2. **Feature extraction** – TF-IDF over unigrams + bigrams, stop-words removed, sublinear term frequency.
3. **Classification** – Naive Bayes, Logistic Regression and Linear SVM are compared; best F1 is kept.
4. **Evaluation** – stratified 80/20 split; accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
5. **Explainability** – LIME perturbs the message (drops words), watches how the probability changes
   and highlights the words that drove the prediction.
6. **Rule-based layer** – regex checks for IP-based links, URL shorteners, odd TLDs and urgency language.

**Why F1/recall and not just accuracy?** Phishing is the minority class, so a model that says "legit" for
everything would still score high accuracy. Recall tells us how many real phishing messages we catch.
""")
