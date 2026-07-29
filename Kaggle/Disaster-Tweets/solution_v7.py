"""v7: pretrained GloVe-Twitter embeddings + mean-pooled feedforward NN.

First deep-learning-family version, moving off the TF-IDF approach that
plateaued at v5/v6. New techniques vs. the classical series:
1. Pretrained word embeddings (GloVe, 100d, trained on Twitter - domain
   match for this dataset) via gensim, instead of TF-IDF bag-of-words.
2. A PyTorch nn.Module: embedding lookup -> mask-aware mean pooling over
   the tweet's tokens -> concat with the 3 scaled numeric features -> small
   dense head -> sigmoid.
3. Manual training loop with early stopping on validation F1 (no sklearn
   Pipeline/GridSearchCV here - different tooling family).

Scope choices, kept deliberately simple as a first NN pass:
- Embeddings are frozen (not fine-tuned) - use GloVe as a fixed feature
  extractor first; fine-tuning is a natural next-version increment.
- Single 80/20 train/val split, not 5-fold CV - full CV around a from-scratch
  training loop is a larger lift, deferred until this pipeline is validated.
- Mean pooling, not an RNN/LSTM/attention - simplest way to turn a variable-
  length token sequence into a fixed-size vector; sequence-aware pooling is
  a natural next step.
"""

import json
import re
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_lookup(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


CONTRACTIONS = load_lookup("english_contractions_lowercase.json")
ACRONYMS = load_lookup("english_acronyms_lowercase.json")


def expand_lookup(text: str, lookup: dict) -> str:
    return " ".join(lookup.get(w, w) for w in text.split())


def clean_text(text: str) -> str:
    # lighter than v3-v6's cleaning: no stopword removal/lemmatization, so
    # tokens stay in a form GloVe's Twitter-trained vocabulary recognizes
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
        embedded = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = torch.cat([pooled, numeric_feats], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # logits


def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            token_ids, numeric = batch[0].to(DEVICE), batch[1].to(DEVICE)
            logits = model(token_ids, numeric)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv"))
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["target"]
    )

    print(f"Loading {EMBEDDING_MODEL_NAME} embeddings (cached after first download)...")
    glove = api.load(EMBEDDING_MODEL_NAME)

    vocab = build_vocab(pd.concat([train_df["text_combined"], test_df["text_combined"]]))
    embedding_matrix = build_embedding_matrix(vocab, glove)
    del glove  # free the 1.19M-word KeyedVectors, only need our slice now

    train_ds = TweetDataset(train_split, vocab, has_target=True)
    val_ds = TweetDataset(val_split, vocab, has_target=True)
    test_ds = TweetDataset(test_df, vocab, has_target=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MeanPoolClassifier(embedding_matrix).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    y_val = val_split["target"].to_numpy()
    best_f1, best_state, epochs_without_improvement = -1, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for token_ids, numeric, target in train_loader:
            token_ids, numeric, target = token_ids.to(DEVICE), numeric.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(token_ids, numeric)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)

        val_proba = predict_proba(model, val_loader)
        val_f1 = f1_score(y_val, (val_proba >= 0.5).astype(int))
        print(f"epoch {epoch:2d}  train_loss={total_loss / len(train_ds):.4f}  val_f1@0.5={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_f1={best_f1:.4f})")
                break

    model.load_state_dict(best_state)

    # threshold tuning on the held-out validation set (only split available here)
    val_proba = predict_proba(model, val_loader)
    best_threshold, best_thresh_f1 = 0.5, f1_score(y_val, (val_proba >= 0.5).astype(int))
    for threshold in np.arange(0.1, 0.91, 0.01):
        f1 = f1_score(y_val, (val_proba >= threshold).astype(int))
        if f1 > best_thresh_f1:
            best_thresh_f1, best_threshold = f1, threshold
    print(f"Best val F1: {best_thresh_f1:.4f} at threshold {best_threshold:.2f}")

    test_proba = predict_proba(model, test_loader)
    test_preds = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v7.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
