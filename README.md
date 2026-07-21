# 📰 Fake News Detection using Machine Learning

An end-to-end ML project: text preprocessing → TF-IDF vectorization → model
training/evaluation → an interactive Streamlit app that predicts whether a
news headline/article is **REAL** or **FAKE**, with a confidence score and
a breakdown of which words drove the prediction.

## Project structure

```
fake-news-detector/
├── app.py                    # Streamlit web app
├── requirements.txt
├── data/
│   ├── generate_dataset.py   # builds the demo dataset (no download needed)
│   └── news_dataset.csv      # generated dataset (text, label)
├── src/
│   ├── preprocess.py         # text cleaning (stopwords, lemmatization)
│   └── train.py              # training + evaluation pipeline
└── models/                   # saved model.joblib / vectorizer.joblib (after training)
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Generate the dataset

This project ships with a **synthetic dataset generator** so you can run the
whole pipeline immediately with zero downloads:

```bash
python data/generate_dataset.py
```

This creates `data/news_dataset.csv` (3,000 balanced REAL/FAKE examples,
built from realistic sensational-vs-factual language templates).

> ⚠️ **Important — read this before putting this on your resume/portfolio:**
> The synthetic data is intentionally template-based so the pipeline works
> instantly, but that also makes it *too easy* — you'll see ~100% accuracy,
> which does not reflect a real-world model. **For a genuinely
> resume-worthy result**, swap in a real dataset before showcasing this
> project (5 minutes of work, see below).

### Using a real dataset instead (recommended)

1. Get the **Kaggle "Fake and Real News Dataset"** — search
   `kaggle fake and real news dataset` (files are usually named
   `Fake.csv` and `True.csv`, ~40,000 rows total).
2. Drop both files into `data/`.
3. Open `src/train.py`, find the `load_data()` function, and replace its
   body with the version shown in the docstring at the top of that
   function (it's already written for you — just uncomment/paste it in).
4. Re-run `python src/train.py`. Expect realistic accuracy in the
   **92–97%** range, which is genuinely impressive and interview-worthy.

## 3. Train the model

```bash
python src/train.py
```

This will:
- Clean and vectorize the text (TF-IDF, unigrams + bigrams, 6,000 features)
- Train **two models** (Logistic Regression and a Passive-Aggressive-style
  linear classifier) and pick the better one by F1 score
- Print a full classification report + confusion matrix
- Save `models/model.joblib`, `models/vectorizer.joblib`, and
  `models/metadata.joblib`

## 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### App features
- Paste any headline or article text
- Get a **REAL / FAKE** prediction with a **confidence percentage**
- See a **word-importance breakdown** — which words in *your* input pushed
  the model toward REAL (green) or FAKE (red), computed straight from the
  model's TF-IDF coefficients (no extra explainability library needed)
- Sidebar shows the model's test-set accuracy/F1
- One-click example buttons to try it instantly

## How it works (for your resume/interview talking points)

1. **Preprocessing** — lowercasing, URL/HTML stripping, punctuation and
   number removal, stopword removal, lemmatization (`src/preprocess.py`).
2. **Vectorization** — `TfidfVectorizer` with unigrams+bigrams turns cleaned
   text into weighted sparse feature vectors.
3. **Modeling** — Logistic Regression and a linear SGD/PA-style classifier
   are trained and compared; the better one (by F1) is kept.
4. **Evaluation** — accuracy, precision, recall, F1, and a confusion matrix
   are printed for the held-out test set.
5. **Explainability** — because the model is linear, each feature has a
   learned coefficient; multiplying by the TF-IDF weight in a specific
   input gives a per-word contribution score — a simple, dependency-free
   form of model explainability.
6. **Deployment** — Streamlit turns the trained artifacts into an
   interactive web app.

## Ideas to extend this further (great for standing out)

- Swap TF-IDF + Logistic Regression for a fine-tuned **DistilBERT**
  (via `transformers`) and compare accuracy/latency trade-offs.
- Add a **source-credibility feature** (e.g. domain reputation) alongside
  text features.
- Deploy the Streamlit app publicly on **Streamlit Community Cloud** and
  link it from your portfolio/LinkedIn/GitHub.
- Add a `/predict` **FastAPI** endpoint so the model can be called from
  other apps, not just the Streamlit UI.
- Track experiments with **MLflow** if you train multiple model variants.
