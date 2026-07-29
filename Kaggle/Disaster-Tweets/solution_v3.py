"""v3: stopword removal + lemmatization, sklearn Pipeline/ColumnTransformer,
k-fold cross-validation, and out-of-fold decision-threshold tuning."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).parent / "data" / "raw"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")

# negation words carry sentiment/meaning - don't let stopword removal drop them
STOPWORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
LEMMATIZER = WordNetLemmatizer()

# best config found in v2's grid search
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 1)
LOGREG_C = 3.0


def load_lookup(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


CONTRACTIONS = load_lookup("english_contractions_lowercase.json")
ACRONYMS = load_lookup("english_acronyms_lowercase.json")


def expand_lookup(text: str, lookup: dict) -> str:
    return " ".join(lookup.get(w, w) for w in text.split())


def remove_stopwords_and_lemmatize(text: str) -> str:
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = expand_lookup(text, CONTRACTIONS)
    text = expand_lookup(text, ACRONYMS)
    text = NON_ALPHA_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    text = remove_stopwords_and_lemmatize(text)
    return text


def clean_keyword(keyword) -> str:
    if pd.isna(keyword):
        return ""
    return keyword.replace("%20", " ").lower()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keyword_clean"] = df["keyword"].apply(clean_keyword)
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_combined"] = (df["keyword_clean"] + " " + df["text_clean"]).str.strip()
    df["has_location"] = df["location"].notna().astype(int)
    df["word_count"] = df["text_clean"].str.split().apply(len)
    df["char_count"] = df["text_clean"].str.len()
    return df


def build_pipeline() -> Pipeline:
    # ColumnTransformer keeps the TF-IDF vectorizer *inside* the pipeline, so
    # cross_val_score/cross_val_predict refit it on each fold's train split
    # only - avoids leaking val-fold vocabulary/IDF stats into training.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=2,
                ),
                "text_combined",
            ),
            ("numeric", "passthrough", ["has_location", "word_count", "char_count"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, C=LOGREG_C)),
        ]
    )


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv"))
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    X = train_df[["text_combined", "has_location", "word_count", "char_count"]]
    y = train_df["target"]

    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. k-fold CV: a mean +/- std F1 estimate is more reliable than one split
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
    print(f"5-fold CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 2. out-of-fold probabilities: each row scored by a model that never saw
    # it in training, so threshold tuning on these isn't optimistically biased
    oof_proba = cross_val_predict(
        pipeline, X, y, cv=skf, method="predict_proba"
    )[:, 1]

    best_threshold, best_f1 = 0.5, f1_score(y, (oof_proba >= 0.5).astype(int))
    for threshold in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y, (oof_proba >= threshold).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    print(f"Best decision threshold: {best_threshold:.2f} (OOF F1={best_f1:.4f})")

    # 3. refit on full training data, predict test with tuned threshold
    pipeline.fit(X, y)
    X_test = test_df[["text_combined", "has_location", "word_count", "char_count"]]
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v3.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
