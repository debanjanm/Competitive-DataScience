"""v2: keyword/location/text-stat features + tuned TF-IDF + LR/SVM/NB ensemble."""

import json
import re
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

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
    return " ".join(lookup.get(w, w) for w in text.split())


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


def clean_keyword(keyword) -> str:
    if pd.isna(keyword):
        return ""
    return keyword.replace("%20", " ").lower()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keyword_clean"] = df["keyword"].apply(clean_keyword)
    df["text_clean"] = df["text"].apply(clean_text)
    # fold keyword into text so TF-IDF vocabulary picks it up directly
    df["text_combined"] = (df["keyword_clean"] + " " + df["text_clean"]).str.strip()
    df["has_location"] = df["location"].notna().astype(int)
    df["word_count"] = df["text_clean"].str.split().apply(len)
    df["char_count"] = df["text_clean"].str.len()
    return df


def numeric_matrix(df: pd.DataFrame) -> csr_matrix:
    return csr_matrix(
        df[["has_location", "word_count", "char_count"]].to_numpy(dtype=float)
    )


def vectorize(text_train, text_val, max_features, ngram_range):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
    Xtr = vec.fit_transform(text_train)
    Xva = vec.transform(text_val)
    return vec, Xtr, Xva


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv"))
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    (
        text_train, text_val,
        y_train, y_val,
        num_train, num_val,
    ) = train_test_split(
        train_df["text_combined"],
        train_df["target"],
        numeric_matrix(train_df),
        test_size=0.2,
        random_state=42,
        stratify=train_df["target"],
    )

    # --- small grid search: TF-IDF shape x LogisticRegression C ---
    best_params, best_f1 = None, -1
    for max_features in (10000, 20000, 30000):
        for ngram_range in ((1, 1), (1, 2)):
            _, Xtr, Xva = vectorize(text_train, text_val, max_features, ngram_range)
            Xtr = hstack([Xtr, num_train]).tocsr()
            Xva = hstack([Xva, num_val]).tocsr()
            for C in (0.5, 1.0, 3.0):
                clf = LogisticRegression(max_iter=1000, C=C)
                clf.fit(Xtr, y_train)
                f1 = f1_score(y_val, clf.predict(Xva))
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"max_features": max_features, "ngram_range": ngram_range, "C": C}

    print(f"Best TF-IDF/LR params: {best_params} -> val F1={best_f1:.4f}")

    vec, Xtr, Xva = vectorize(
        text_train, text_val, best_params["max_features"], best_params["ngram_range"]
    )
    Xtr = hstack([Xtr, num_train]).tocsr()
    Xva = hstack([Xva, num_val]).tocsr()

    models = {
        "logreg": LogisticRegression(max_iter=1000, C=best_params["C"]),
        "linear_svc": LinearSVC(C=1.0),
        "complement_nb": ComplementNB(),
    }
    for name, model in models.items():
        model.fit(Xtr, y_train)
        f1 = f1_score(y_val, model.predict(Xva))
        print(f"{name} val F1={f1:.4f}")

    ensemble = VotingClassifier(estimators=list(models.items()), voting="hard")
    ensemble.fit(Xtr, y_train)
    ensemble_f1 = f1_score(y_val, ensemble.predict(Xva))
    print(f"ensemble (hard vote) val F1={ensemble_f1:.4f}")

    # pick whichever scored best on validation for the final fit
    candidates = {**models, "ensemble": ensemble}
    scores = {
        name: f1_score(y_val, m.predict(Xva)) for name, m in candidates.items()
    }
    best_name = max(scores, key=scores.get)
    print(f"Selected final model: {best_name} (val F1={scores[best_name]:.4f})")

    # --- refit on full training data with chosen model + vectorizer config ---
    final_vec = TfidfVectorizer(
        max_features=best_params["max_features"],
        ngram_range=best_params["ngram_range"],
        min_df=2,
    )
    X_full = final_vec.fit_transform(train_df["text_combined"])
    X_full = hstack([X_full, numeric_matrix(train_df)]).tocsr()
    X_test = final_vec.transform(test_df["text_combined"])
    X_test = hstack([X_test, numeric_matrix(test_df)]).tocsr()

    if best_name == "ensemble":
        final_model = VotingClassifier(
            estimators=[
                ("logreg", LogisticRegression(max_iter=1000, C=best_params["C"])),
                ("linear_svc", LinearSVC(C=1.0)),
                ("complement_nb", ComplementNB()),
            ],
            voting="hard",
        )
    elif best_name == "logreg":
        final_model = LogisticRegression(max_iter=1000, C=best_params["C"])
    elif best_name == "linear_svc":
        final_model = LinearSVC(C=1.0)
    else:
        final_model = ComplementNB()

    final_model.fit(X_full, train_df["target"])
    test_preds = final_model.predict(X_test)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v2.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
