"""
train.py
--------
End-to-end training pipeline:
  1. Load data (synthetic dataset by default; see README to swap in a real
     Kaggle dataset).
  2. Clean text (src/preprocess.py).
  3. Vectorize with TF-IDF.
  4. Train + compare two models (Logistic Regression, Passive Aggressive).
  5. Evaluate (accuracy, precision, recall, F1, confusion matrix).
  6. Save the best model + vectorizer to models/ with joblib.
"""

import os
import sys
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "news_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """
    Loads data/news_dataset.csv (text,label columns; label 1=REAL, 0=FAKE).

    TO USE A REAL DATASET INSTEAD (recommended for a resume-grade model):
    Download the Kaggle "Fake and Real News Dataset" (Fake.csv + True.csv),
    place both files in data/, and replace the body of this function with:

        fake = pd.read_csv(os.path.join(BASE_DIR, "data", "Fake.csv"))
        real = pd.read_csv(os.path.join(BASE_DIR, "data", "True.csv"))
        fake["label"] = 0
        real["label"] = 1
        df = pd.concat([fake, real], ignore_index=True)
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_dataset.py` first, "
            "or drop in a real dataset as described in the README."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} rows  |  REAL={sum(df.label == 1)}  FAKE={sum(df.label == 0)}")

    # 2. Clean
    print("\n[2/5] Cleaning text...")
    t0 = time.time()
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"    Done in {time.time() - t0:.2f}s")

    # 3. Split + Vectorize
    print("\n[3/5] Splitting and vectorizing (TF-IDF)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")

    # 4. Train + compare models
    print("\n[4/5] Training models...")
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=5),
        "PassiveAggressive": SGDClassifier(
            loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
            max_iter=1000, random_state=42,
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"model": model, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "preds": preds}
        print(f"    {name:20s}  acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print(f"\n    -> Best model: {best_name} (F1={best['f1']:.4f})")

    # 5. Full evaluation report + save
    print("\n[5/5] Final evaluation for best model:")
    print(classification_report(y_test, best["preds"], target_names=["FAKE", "REAL"]))
    cm = confusion_matrix(y_test, best["preds"])
    print("Confusion matrix (rows=actual, cols=predicted) [FAKE, REAL]:")
    print(cm)

    model_path = os.path.join(MODEL_DIR, "model.joblib")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    joblib.dump(best["model"], model_path)
    joblib.dump(vectorizer, vec_path)

    meta_path = os.path.join(MODEL_DIR, "metadata.joblib")
    joblib.dump(
        {
            "model_name": best_name,
            "accuracy": best["acc"],
            "f1": best["f1"],
            "precision": best["prec"],
            "recall": best["rec"],
            "confusion_matrix": cm.tolist(),
        },
        meta_path,
    )

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved vectorizer -> {vec_path}")
    print(f"Saved metadata   -> {meta_path}")
    print("\nDone! Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
