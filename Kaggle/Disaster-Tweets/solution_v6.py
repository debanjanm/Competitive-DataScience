"""v6: TruncatedSVD (LSA) dimensionality reduction + class_weight tuning.

Last classical-ML round before moving to embeddings/deep learning (per
v5's Next Steps). Two changes on top of v5's word+char TF-IDF + scaled
numeric features setup:
1. Each TF-IDF block is followed by TruncatedSVD (LSA) - collapses the
   sparse ~30000-dim word+char TF-IDF space into a small dense space of
   latent semantic components, which also lets word/char/numeric blocks
   sit on comparable dense footing instead of two huge sparse blocks next
   to three scaled scalars.
2. class_weight is searched (None vs "balanced") - train target is ~57/43,
   a mild imbalance never previously checked.

Train target balance: ~57% class 0, ~43% class 1 (checked once, not
re-derived in this script - see SOLUTION_v6.md).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data" / "raw"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")

STOPWORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
LEMMATIZER = WordNetLemmatizer()

WORD_TFIDF_MAX_FEATURES = 10000
WORD_TFIDF_NGRAM_RANGE = (1, 2)
CHAR_TFIDF_MAX_FEATURES = 20000
CHAR_TFIDF_NGRAM_RANGE = (3, 5)
SVD_RANDOM_STATE = 42


def load_lookup(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


CONTRACTIONS = load_lookup("english_contractions_lowercase.json")
ACRONYMS = load_lookup("english_acronyms_lowercase.json")


def expand_lookup(text: str, lookup: dict) -> str:
    return " ".join(lookup.get(w, w) for w in text.split())


def to_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def remove_stopwords_and_lemmatize(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    tagged = pos_tag(tokens)
    lemmas = [
        LEMMATIZER.lemmatize(token, to_wordnet_pos(tag))
        for token, tag in tagged
        if token not in STOPWORDS
    ]
    return " ".join(lemmas)


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
    word_block = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=WORD_TFIDF_MAX_FEATURES,
                    ngram_range=WORD_TFIDF_NGRAM_RANGE,
                    min_df=2,
                ),
            ),
            ("svd", TruncatedSVD(n_components=100, random_state=SVD_RANDOM_STATE)),
        ]
    )
    char_block = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=CHAR_TFIDF_MAX_FEATURES,
                    ngram_range=CHAR_TFIDF_NGRAM_RANGE,
                    min_df=2,
                ),
            ),
            ("svd", TruncatedSVD(n_components=100, random_state=SVD_RANDOM_STATE)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("word_tfidf", word_block, "text_combined"),
            ("char_tfidf", char_block, "text_combined"),
            (
                "numeric",
                StandardScaler(),
                ["has_location", "word_count", "char_count"],
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv"))
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    X = train_df[["text_combined", "has_location", "word_count", "char_count"]]
    y = train_df["target"]

    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TF-IDF vectorizer params fixed at v5's winners; search SVD rank,
    # class_weight, and C to keep the grid small
    param_grid = {
        "preprocess__word_tfidf__svd__n_components": [100, 200],
        "clf__class_weight": [None, "balanced"],
        "clf__C": [0.5, 1.0, 3.0],
    }
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=skf, n_jobs=-1)
    search.fit(X, y)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")

    best_pipeline = search.best_estimator_

    tuning_pipeline = build_pipeline()
    tuning_pipeline.set_params(**search.best_params_)
    oof_proba = cross_val_predict(
        tuning_pipeline, X, y, cv=skf, method="predict_proba"
    )[:, 1]

    best_threshold, best_f1 = 0.5, f1_score(y, (oof_proba >= 0.5).astype(int))
    for threshold in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y, (oof_proba >= threshold).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    print(f"Best decision threshold: {best_threshold:.2f} (OOF F1={best_f1:.4f})")

    X_test = test_df[["text_combined", "has_location", "word_count", "char_count"]]
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v6.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
