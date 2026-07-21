"""
app.py
------
Streamlit UI for the Fake News Detector.

Run with:
    streamlit run app.py

Features:
  - Paste a headline or full article.
  - Get a REAL / FAKE prediction with a confidence score.
  - See the words that pushed the prediction toward REAL or FAKE
    (computed directly from the linear model's TF-IDF coefficients —
    no extra explainability library required).
  - Clean, interactive UI with example buttons and a confidence gauge.
"""

import os
import sys
import numpy as np
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.joblib")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


# ---------------------------------------------------------------------------
# Load model artifacts (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
    return model, vectorizer, meta


model, vectorizer, meta = load_artifacts()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(text: str):
    """Returns (label, confidence 0-1, top_words list of (word, weight))."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]  # [P(FAKE), P(REAL)]
        pred = int(np.argmax(proba))
        confidence = float(proba[pred])
    else:
        # SGD/PassiveAggressive-style models: use decision_function -> sigmoid
        score = model.decision_function(vec)[0]
        pred = int(score > 0)
        confidence = float(sigmoid(abs(score)))

    label = "REAL" if pred == 1 else "FAKE"

    # Word-level influence: coefficient * tfidf value for words present in this text
    top_words = []
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        nonzero_idx = vec.nonzero()[1]
        contributions = [(feature_names[i], coefs[i] * vec[0, i]) for i in nonzero_idx]
        # Sort by absolute contribution, keep the most influential words
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_words = contributions[:10]

    return label, confidence, top_words


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption("TF-IDF + Machine Learning · paste a headline or article to check it")

if model is None:
    st.error(
        "No trained model found. Run these commands first, then reload this app:\n\n"
        "```\npython data/generate_dataset.py\npython src/train.py\n```"
    )
    st.stop()

with st.sidebar:
    st.header("ℹ️ Model info")
    if meta:
        st.metric("Test accuracy", f"{meta.get('accuracy', 0) * 100:.1f}%")
        st.metric("F1 score", f"{meta.get('f1', 0):.3f}")
        st.write(f"**Model:** {meta.get('model_name', 'Unknown')}")
    st.divider()
    st.caption(
        "⚠️ This demo is trained on a synthetic dataset generated for this "
        "project. Swap in a real dataset (see README) before using this for "
        "anything beyond a portfolio demo."
    )

st.subheader("Try an example")
examples = {
    "Sensational example": (
        "SHOCKING: the economy will completely change overnight, insiders claim. "
        "Share this before it gets DELETED!!! Mainstream media REFUSES to cover this story."
    ),
    "Factual example": (
        "Reuters reported that the housing market saw a 3.2% change over the past quarter, "
        "according to figures released on Tuesday."
    ),
}
col1, col2 = st.columns(2)
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if col1.button("Load sensational example", use_container_width=True):
    st.session_state.text_input = examples["Sensational example"]
if col2.button("Load factual example", use_container_width=True):
    st.session_state.text_input = examples["Factual example"]

text_input = st.text_area(
    "Paste a news headline or article:",
    height=180,
    key="text_input",
    placeholder="e.g. Officials announced today that...",
)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        label, confidence, top_words = predict(text_input)

        st.divider()
        if label == "REAL":
            st.success(f"### ✅ Prediction: REAL  \nConfidence: {confidence * 100:.1f}%")
        else:
            st.error(f"### 🚩 Prediction: FAKE  \nConfidence: {confidence * 100:.1f}%")

        st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

        if top_words:
            st.subheader("Words influencing this prediction")
            st.caption("Green pushes toward REAL, red pushes toward FAKE")

            max_abs = max(abs(w) for _, w in top_words) or 1.0
            for word, weight in top_words:
                bar_width = int(abs(weight) / max_abs * 100)
                color = "#22c55e" if weight > 0 else "#ef4444"
                direction = "REAL" if weight > 0 else "FAKE"
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;margin-bottom:6px;">
                        <div style="width:110px;font-family:monospace;">{word}</div>
                        <div style="background:{color};height:14px;width:{bar_width}%;
                                    border-radius:4px;margin-right:8px;"></div>
                        <div style="font-size:12px;color:#888;">→ {direction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No overlapping vocabulary words were found to explain this prediction.")

st.divider()
st.caption("Built with ❤️ by Niha Hawas using Python · Pandas · Scikit-learn · Streamlit")
