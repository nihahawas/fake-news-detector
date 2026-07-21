"""
app.py
------
Streamlit UI for the Fake News Detector — "The Verification Desk" theme.

Run with:
    streamlit run app.py

Features:
  - Paste a headline or full article ("submission slip").
  - Get a REAL / FAKE verdict, shown as a stamped verification card.
  - See a wire-evidence breakdown of which words drove the call,
    computed directly from the linear model's TF-IDF coefficients —
    no extra explainability library required.
  - Press-credentials sidebar with the model's test performance.
"""

import os
import sys
import html
import numpy as np
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.joblib")

st.set_page_config(page_title="The Verification Desk", page_icon="🗞️", layout="centered")


# ---------------------------------------------------------------------------
# Theme — "The Verification Desk"
# Deep ink-navy desk, aged-paper cards, brass rule lines, and a rotated
# rubber-stamp verdict (VERIFIED / FLAGGED) as the one bold signature move.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,900&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #101a30;
        --ink-deep: #0a1224;
        --paper: #ece4d0;
        --paper-line: #d6cbac;
        --gold: #c9a227;
        --verified: #1f7a4d;
        --flagged: #ad3a2e;
        --text-ink: #241f14;
        --muted: #8c8570;
    }

    .stApp { background: var(--ink); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 2.2rem; max-width: 760px; }

    /* ---- Masthead ---- */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.18em;
        font-size: 0.72rem;
        color: var(--gold);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .masthead-rule { border: none; border-top: 1px solid rgba(201,162,39,0.45); margin: 0.6rem 0 1.1rem 0; }
    .masthead-title {
        font-family: 'Fraunces', serif;
        font-weight: 900;
        font-optical-sizing: auto;
        font-size: 2.6rem;
        line-height: 1.05;
        color: var(--paper);
        margin: 0 0 0.35rem 0;
    }
    .masthead-sub {
        color: #9aa3b8;
        font-size: 0.98rem;
        margin-bottom: 0.2rem;
    }

    /* ---- Section labels ---- */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.14em;
        font-size: 0.72rem;
        text-transform: uppercase;
        color: var(--gold);
        margin: 1.6rem 0 0.5rem 0;
    }

    /* ---- Inputs restyled as a paper "submission slip" ---- */
    [data-testid="stTextArea"] textarea {
        background: var(--paper) !important;
        color: var(--text-ink) !important;
        border: 1px solid var(--paper-line) !important;
        border-radius: 3px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.98rem !important;
        padding: 1rem !important;
    }
    [data-testid="stTextArea"] textarea::placeholder { color: var(--muted) !important; }

    /* ---- Buttons ---- */
    .stButton > button {
        background: var(--ink-deep);
        color: var(--paper);
        border: 1px solid rgba(201,162,39,0.5);
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        font-size: 0.82rem;
        text-transform: uppercase;
        padding: 0.55rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--gold);
        color: var(--gold);
    }
    .stButton > button[kind="primary"] {
        background: var(--gold);
        color: var(--ink-deep);
        border: 1px solid var(--gold);
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: #ddb437;
        color: var(--ink-deep);
    }

    /* ---- Sidebar: press credentials ---- */
    [data-testid="stSidebar"] {
        background: var(--ink-deep);
        border-right: 1px solid rgba(201,162,39,0.25);
    }
    [data-testid="stSidebar"] .eyebrow { color: var(--gold); }
    [data-testid="stMetric"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #9aa3b8 !important;
        text-transform: uppercase;
        font-size: 0.68rem !important;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif !important;
        color: var(--paper) !important;
    }

    /* ---- Verdict card ---- */
    .verdict-card {
        position: relative;
        background: var(--paper);
        border: 1px solid var(--paper-line);
        border-radius: 4px;
        padding: 1.6rem 1.7rem 1.4rem 1.7rem;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .stamp {
        position: absolute;
        top: 18px;
        right: -8px;
        font-family: 'Fraunces', serif;
        font-weight: 900;
        font-size: 1.55rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 6px 18px;
        border: 4px double currentColor;
        border-radius: 6px;
        transform: rotate(-8deg);
        opacity: 0.9;
        mix-blend-mode: multiply;
    }
    .stamp.verified { color: var(--verified); }
    .stamp.flagged  { color: var(--flagged); }

    .verdict-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.15rem;
    }
    .verdict-word {
        font-family: 'Fraunces', serif;
        font-weight: 900;
        font-size: 1.9rem;
        margin-bottom: 0.7rem;
    }
    .verdict-word.verified { color: var(--verified); }
    .verdict-word.flagged  { color: var(--flagged); }

    /* ---- Confidence meter ---- */
    .meter-row { display: flex; align-items: center; gap: 10px; margin-top: 0.3rem; }
    .meter-track {
        flex: 1;
        height: 8px;
        background: rgba(0,0,0,0.12);
        border-radius: 4px;
        overflow: hidden;
    }
    .meter-fill { height: 100%; border-radius: 4px; }
    .meter-fill.verified { background: var(--verified); }
    .meter-fill.flagged  { background: var(--flagged); }
    .meter-pct {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-ink);
        min-width: 48px;
        text-align: right;
    }

    /* ---- Wire evidence list ---- */
    .evidence-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.3rem 0 0.6rem 0;
        border-top: 1px dashed var(--paper-line);
        padding-top: 0.9rem;
    }
    .evidence-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 5px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
    .evidence-word { width: 130px; color: var(--text-ink); flex-shrink: 0; }
    .evidence-track { flex: 1; height: 10px; background: rgba(0,0,0,0.06); border-radius: 2px; }
    .evidence-fill { height: 100%; border-radius: 2px; }
    .evidence-dir { width: 60px; font-size: 0.72rem; color: var(--muted); text-align: right; }

    .fine-print {
        font-size: 0.78rem;
        color: #7c8499;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        score = model.decision_function(vec)[0]
        pred = int(score > 0)
        confidence = float(sigmoid(abs(score)))

    label = "REAL" if pred == 1 else "FAKE"

    top_words = []
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        nonzero_idx = vec.nonzero()[1]
        contributions = [(feature_names[i], coefs[i] * vec[0, i]) for i in nonzero_idx]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_words = contributions[:10]

    return label, confidence, top_words


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="eyebrow">Issue No. 001 &nbsp;·&nbsp; AI Verification Desk</div>
    <div class="masthead-title">The Verification Desk</div>
    <div class="masthead-sub">Submit a clipping. We check it against the wire.</div>
    <hr class="masthead-rule" />
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.markdown(
        '<div class="verdict-card">'
        '<div class="verdict-label">Desk notice</div>'
        '<div class="verdict-word" style="color:var(--flagged);font-size:1.2rem;">No trained model on file</div>'
        '<div class="fine-print" style="color:var(--text-ink);">Run these two commands, then reload this page:<br>'
        '<code>python data/generate_dataset.py</code><br><code>python src/train.py</code></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — press credentials
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Press Credentials</div>', unsafe_allow_html=True)
    if meta:
        st.metric("Test accuracy", f"{meta.get('accuracy', 0) * 100:.1f}%")
        st.metric("F1 score", f"{meta.get('f1', 0):.3f}")
        st.markdown(
            f'<div class="fine-print" style="margin-top:0.4rem;">Model on desk: '
            f'<strong style="color:var(--paper);">{meta.get("model_name", "Unknown")}</strong></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr class="masthead-rule" />', unsafe_allow_html=True)
    st.markdown(
        '<div class="fine-print">This desk is staffed by a model trained on a '
        'synthetic wire generated for this project. Swap in a real dataset '
        '(see README) before running this beyond a portfolio demo.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Sample clippings</div>', unsafe_allow_html=True)
examples = {
    "sensational": (
        "SHOCKING: the economy will completely change overnight, insiders claim. "
        "Share this before it gets DELETED!!! Mainstream media REFUSES to cover this story."
    ),
    "factual": (
        "Reuters reported that the housing market saw a 3.2% change over the past quarter, "
        "according to figures released on Tuesday."
    ),
}
col1, col2 = st.columns(2)
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if col1.button("Load a known fake clip", use_container_width=True):
    st.session_state.text_input = examples["sensational"]
if col2.button("Load a known real clip", use_container_width=True):
    st.session_state.text_input = examples["factual"]

# ---------------------------------------------------------------------------
# Submission slip
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Submission slip</div>', unsafe_allow_html=True)
text_input = st.text_area(
    " ",
    height=170,
    key="text_input",
    placeholder="Paste a news headline or article for the desk to check…",
    label_visibility="collapsed",
)

analyze = st.button("Analyze clipping", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
if analyze:
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        label, confidence, top_words = predict(text_input)
        css_class = "verified" if label == "REAL" else "flagged"
        stamp_word = "VERIFIED" if label == "REAL" else "FLAGGED"

        evidence_html = ""
        if top_words:
            max_abs = max(abs(w) for _, w in top_words) or 1.0
            for word, weight in top_words:
                bar_width = max(6, int(abs(weight) / max_abs * 100))
                direction_class = "verified" if weight > 0 else "flagged"
                direction_label = "→ real" if weight > 0 else "→ fake"
                evidence_html += (
                    f'<div class="evidence-row">'
                    f'<div class="evidence-word">{html.escape(word)}</div>'
                    f'<div class="evidence-track"><div class="evidence-fill {direction_class}" '
                    f'style="width:{bar_width}%;"></div></div>'
                    f'<div class="evidence-dir">{direction_label}</div>'
                    f"</div>"
                )
        else:
            evidence_html = (
                '<div class="fine-print" style="color:var(--text-ink);">'
                "No overlapping wire vocabulary was found to explain this call.</div>"
            )

        st.markdown(
            f"""
            <div class="verdict-card">
                <div class="stamp {css_class}">{stamp_word}</div>
                <div class="verdict-label">Desk verdict</div>
                <div class="verdict-word {css_class}">{label}</div>
                <div class="meter-row">
                    <div class="meter-track">
                        <div class="meter-fill {css_class}" style="width:{confidence*100:.0f}%;"></div>
                    </div>
                    <div class="meter-pct">{confidence*100:.0f}%</div>
                </div>
                <div class="evidence-title">Wire evidence — words that swayed the call</div>
                {evidence_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="fine-print" style="margin-top:2rem;text-align:center;">'
    "Built with Python · Pandas · Scikit-learn · Streamlit</div>",
    unsafe_allow_html=True,
)
