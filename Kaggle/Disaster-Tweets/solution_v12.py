"""v12: wrap v11's DistilBERT fine-tune in 5-fold CV, closing the same
eval-rigor gap v8 closed for the GloVe+NN series (v7 -> v8 pattern).

Same model/tokenization/training loop as v11 - only the evaluation
protocol changes:
- StratifiedKFold(5) instead of one train_test_split.
- Each fold fine-tunes a *fresh* DistilBERT (re-loaded from the pretrained
  checkpoint) - reusing one model across folds would leak fold 1's
  fine-tuning into fold 2's "held-out" evaluation.
- Out-of-fold probabilities collected across all 5 folds for honest
  threshold tuning, same technique as v8/v9/v10.
- Test predictions = average of all 5 fold models' probabilities.

Cost note: this is ~5x v11's ~37-minute run, since each fold repeats the
full fine-tune from the pretrained checkpoint. Expect ~2.5-3+ hours total.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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
N_FOLDS = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def light_clean(text: str) -> str:
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


def train_one_fold(train_split, val_split, tokenizer, test_ds, test_loader, fold_idx):
    # fresh pretrained checkpoint per fold - no state carried over between folds
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = TweetDataset(train_split["text_combined"], tokenizer, train_split["target"].to_numpy())
    val_ds = TweetDataset(val_split["text_combined"], tokenizer, val_split["target"].to_numpy())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    y_val = val_split["target"].to_numpy()
    best_f1, best_state, epochs_without_improvement = -1, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
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
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return fold_val_proba, fold_test_proba


def main():
    train_df = build_features(pd.read_csv(DATA_DIR / "train.csv")).reset_index(drop=True)
    test_df = build_features(pd.read_csv(DATA_DIR / "test.csv"))

    print(f"Loading {MODEL_NAME} tokenizer (cached after first download)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_ds = TweetDataset(test_df["text_combined"], tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    y = train_df["target"].to_numpy()
    oof_proba = np.zeros(len(train_df))
    test_proba_folds = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y), start=1):
        train_split = train_df.iloc[train_idx]
        val_split = train_df.iloc[val_idx]
        fold_val_proba, fold_test_proba = train_one_fold(
            train_split, val_split, tokenizer, test_ds, test_loader, fold_idx
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

    test_proba = np.mean(test_proba_folds, axis=0)
    test_preds = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission_path = Path(__file__).parent / "submission_v12.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


if __name__ == "__main__":
    main()
