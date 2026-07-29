"""v11: fine-tuned DistilBERT - the "big jump" step flagged since v1's
Approach Notes, taken now that TF-IDF (v1-v6) and GloVe+NN (v7-v10) both
plateaued around F1 0.76-0.78.

Genuinely different technique family from everything before it:
- Contextual embeddings, not static ones - the same word gets a different
  vector depending on its sentence (v7-v10's GloVe gave "fire" one fixed
  vector regardless of context; DistilBERT's self-attention lets the model
  distinguish "the sky is on fire" from "you're fired").
- Subword WordPiece tokenization, not whole-word vocab lookup - no OOV
  problem the way v7-v10 had 13% of tokens fall back to random vectors.
- Minimal text preprocessing on purpose - BERT-family models are pretrained
  on naturally-written text, so heavy cleaning (stopword removal,
  lemmatization) actively works against what the tokenizer/model expect.
  Only URLs/mentions are stripped here; casing and punctuation are left for
  the (uncased) tokenizer to handle itself.
- Single 80/20 split, not 5-fold CV - fine-tuning even DistilBERT 5x is a
  much bigger time cost than the sklearn or GloVe+NN cases; deferred for
  the same reason v7 deferred CV before v8 added it.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

DATA_DIR = Path(__file__).parent / "data" / "raw"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 4
PATIENCE = 2
LEARNING_RATE = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def light_clean(text: str) -> str:
    # deliberately minimal - see module docstring
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def clean_keyword(keyword) -> str:
    if pd.isna(keyword):
        return ""
    return keyword.replace("%20", " ")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keyword_clean"] = df["keyword"].apply(clean_keyword)
    df["text_clean"] = df["text"].apply(light_clean)
    df["text_combined"] = (df["keyword_clean"] + " " + df["text_clean"]).str.strip()
    return df


class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, targets=None):
        encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.targets = torch.tensor(targets, dtype=torch.long) if targets is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = (self.input_ids[idx], self.attention_mask[idx])
        if self.targets is not None:
            return item + (self.targets[idx],)
        return item


def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
    return np.concatenate(probs)


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv"))
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["target"]
    )

    print(f"Loading {MODEL_NAME} tokenizer + weights (cached after first download)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = TweetDataset(train_split["text_combined"], tokenizer, train_split["target"].to_numpy())
    val_ds = TweetDataset(val_split["text_combined"], tokenizer, val_split["target"].to_numpy())
    test_ds = TweetDataset(test_df["text_combined"], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    y_val = val_split["target"].to_numpy()
    best_f1, best_state, epochs_without_improvement = -1, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, target in train_loader:
            input_ids, attention_mask, target = (
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                target.to(DEVICE),
            )
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item() * len(target)

        val_proba = predict_proba(model, val_loader)
        val_f1 = f1_score(y_val, (val_proba >= 0.5).astype(int))
        print(f"epoch {epoch}  train_loss={total_loss / len(train_ds):.4f}  val_f1@0.5={val_f1:.4f}")

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
    submission_path = Path(__file__).parent / "submission_v11.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
