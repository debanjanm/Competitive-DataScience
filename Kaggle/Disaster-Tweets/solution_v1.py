"""Baseline: TF-IDF + Logistic Regression for disaster tweet classification."""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data" / "raw"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def load_lookup(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


CONTRACTIONS = load_lookup("english_contractions_lowercase.json")
ACRONYMS = load_lookup("english_acronyms_lowercase.json")


def expand_lookup(text: str, lookup: dict) -> str:
    words = text.split()
    return " ".join(lookup.get(w, w) for w in words)


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = expand_lookup(text, CONTRACTIONS)
    text = expand_lookup(text, ACRONYMS)
    text = NON_ALPHA_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text_clean"],
        train_df["target"],
        test_size=0.2,
        random_state=42,
        stratify=train_df["target"],
    )

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    val_preds = model.predict(X_val_vec)
    val_f1 = f1_score(y_val, val_preds)
    print(f"Validation F1: {val_f1:.4f}")

    # Refit on full training data before predicting on test set
    X_full_vec = vectorizer.fit_transform(train_df["text_clean"])
    model.fit(X_full_vec, train_df["target"])

    X_test_vec = vectorizer.transform(test_df["text_clean"])
    test_preds = model.predict(X_test_vec)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v1.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
