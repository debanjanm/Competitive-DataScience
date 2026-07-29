"""v8: wrap v7's GloVe + mean-pool NN in 5-fold CV, closing the eval-rigor
gap v7 explicitly deferred (single 80/20 split there wasn't comparable to
v3-v6's 5-fold CV means).

Same model/features/training loop as v7 - only the evaluation protocol
changes:
- StratifiedKFold(5) instead of one train_test_split.
- Each fold trains its own model (early stopping on that fold's val F1).
- Out-of-fold probabilities (each row scored by a model that never trained
  on it) are collected across all 5 folds for honest threshold tuning -
  same OOF technique used in v3-v6, just driven by a manual loop instead of
  sklearn's cross_val_predict.
- Test predictions are the *average* of all 5 fold models' probabilities
  (a bagging-style ensemble) rather than one final refit - standard practice
  for k-fold neural net training, and cheaper than a separate full-data
  refit pass.
"""

import json
import re
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent / "data" / "raw"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")

EMBEDDING_MODEL_NAME = "glove-twitter-100"
EMBED_DIM = 100
MAX_LEN = 32
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 4
LEARNING_RATE = 1e-3
N_FOLDS = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
    df["text_combined"] = (df["keyword_clean"] + " " + df["text_clean"]).str.strip()
    df["has_location"] = df["location"].notna().astype(int)
    df["word_count"] = df["text_clean"].str.split().apply(len)
    df["char_count"] = df["text_clean"].str.len()
    return df


def build_vocab(texts: pd.Series) -> dict:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_embedding_matrix(vocab: dict, glove) -> np.ndarray:
    matrix = np.random.normal(scale=0.1, size=(len(vocab), EMBED_DIM)).astype(np.float32)
    matrix[0] = 0.0  # <pad>
    hits = 0
    for token, idx in vocab.items():
        if token in glove:
            matrix[idx] = glove[token]
            hits += 1
    print(f"Embedding coverage: {hits}/{len(vocab)} tokens found in GloVe ({hits / len(vocab):.1%})")
    return matrix


def encode(text: str, vocab: dict) -> list:
    ids = [vocab.get(tok, 1) for tok in text.split()[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return ids


class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict, has_target: bool):
        self.token_ids = np.array([encode(t, vocab) for t in df["text_combined"]], dtype=np.int64)
        numeric = df[["has_location", "word_count", "char_count"]].to_numpy(dtype=np.float32)
        self.numeric = (numeric - numeric.mean(axis=0)) / (numeric.std(axis=0) + 1e-6)
        self.targets = df["target"].to_numpy(dtype=np.float32) if has_target else None

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        item = (torch.tensor(self.token_ids[idx]), torch.tensor(self.numeric[idx]))
        if self.targets is not None:
            return item + (torch.tensor(self.targets[idx]),)
        return item


class MeanPoolClassifier(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, numeric_dim: int = 3, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=True, padding_idx=0
        )
        embed_dim = embedding_matrix.shape[1]
        self.fc1 = nn.Linear(embed_dim + numeric_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, token_ids, numeric_feats):
        embedded = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = torch.cat([pooled, numeric_feats], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            token_ids, numeric = batch[0].to(DEVICE), batch[1].to(DEVICE)
            logits = model(token_ids, numeric)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_one_fold(train_split, val_split, embedding_matrix, vocab, test_ds, test_loader, fold_idx):
    train_ds = TweetDataset(train_split, vocab, has_target=True)
    val_ds = TweetDataset(val_split, vocab, has_target=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MeanPoolClassifier(embedding_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    y_val = val_split["target"].to_numpy()
    best_f1, best_state, epochs_without_improvement = -1, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for token_ids, numeric, target in train_loader:
            token_ids, numeric, target = token_ids.to(DEVICE), numeric.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(token_ids, numeric), target)
            loss.backward()
            optimizer.step()

        val_proba = predict_proba(model, val_loader)
        val_f1 = f1_score(y_val, (val_proba >= 0.5).astype(int))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    print(f"fold {fold_idx}  best val_f1@0.5={best_f1:.4f}")

    fold_val_proba = predict_proba(model, val_loader)
    fold_test_proba = predict_proba(model, test_loader)
    return fold_val_proba, fold_test_proba


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv")).reset_index(drop=True)
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    print(f"Loading {EMBEDDING_MODEL_NAME} embeddings (cached after first download)...")
    glove = api.load(EMBEDDING_MODEL_NAME)
    vocab = build_vocab(pd.concat([train_df["text_combined"], test_df["text_combined"]]))
    embedding_matrix = build_embedding_matrix(vocab, glove)
    del glove

    test_ds = TweetDataset(test_df, vocab, has_target=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    y = train_df["target"].to_numpy()
    oof_proba = np.zeros(len(train_df))
    test_proba_folds = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        fold_val_proba, fold_test_proba = train_one_fold(
            train_split, val_split, embedding_matrix, vocab, test_ds, test_loader, fold_idx
        )
        oof_proba[val_idx] = fold_val_proba
        test_proba_folds.append(fold_test_proba)

    oof_f1_at_05 = f1_score(y, (oof_proba >= 0.5).astype(int))
    print(f"5-fold OOF F1 @ threshold 0.5: {oof_f1_at_05:.4f}")

    best_threshold, best_f1 = 0.5, oof_f1_at_05
    for threshold in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y, (oof_proba >= threshold).astype(int))
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    print(f"Best decision threshold: {best_threshold:.2f} (OOF F1={best_f1:.4f})")

    # average the 5 fold models' test predictions - bagging-style ensemble,
    # cheaper than a separate full-data refit and uses every trained model
    test_proba = np.mean(test_proba_folds, axis=0)
    test_preds = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v8.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
