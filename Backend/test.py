import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import MusicDataset
from model import ModelConfig, MultiTaskMultimodalLSTM
from genre_mapper import MAIN_GENRES

# =========================
# CONFIG
# =========================
CSV_FILE = "../data/labels_test.csv"
AUDIO_DIR = "../data/audio"
LYRICS_DIR = "../data/lyrics"
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Tokenizer (same as training)
# =========================
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]

# =========================
# Collate Function (CRITICAL)
# =========================
def collate_fn(batch):
    batch = [x for x in batch if x is not None and "genre" in x]
    if len(batch) == 0:
        return None

    mel_lens = [item["mel_len"] for item in batch]
    mel_padded = pad_sequence(
        [item["mel"] for item in batch],
        batch_first=True
    )

    tok_lens = [item["tok_len"] for item in batch]
    tokens_padded = pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=0
    )

    return {
        "mel": mel_padded,
        "mel_lens": torch.tensor(mel_lens),
        "tokens": tokens_padded,
        "tok_lens": torch.tensor(tok_lens),
        "valence": torch.stack([item["valence"] for item in batch]),
        "arousal": torch.stack([item["arousal"] for item in batch]),
        "genre": torch.stack([item["genre"] for item in batch]),
    }

# =========================
# Load Dataset
# =========================
print("📄 Loading test dataset...")

genre_map = {g: i for i, g in enumerate(MAIN_GENRES)}

test_dataset = MusicDataset(
    CSV_FILE,
    AUDIO_DIR,
    LYRICS_DIR,
    genre_map=genre_map,
    tokenizer=simple_tokenizer
)

print(f"🧪 Testing samples: {len(test_dataset)}")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# Load Model
# =========================
cfg = ModelConfig(
    n_mels=80,
    vocab_size=256,
    pad_idx=0,
    n_genre=len(MAIN_GENRES),
    hidden_size=64,
    num_layers=2
)

model = MultiTaskMultimodalLSTM(cfg).to(DEVICE)
model.load_state_dict(torch.load("Save_model/best_model.pth", map_location=DEVICE))
model.eval()

print("✅ Model loaded successfully")

# =========================
# Testing Loop
# =========================
all_preds = []
all_targets = []

valence_errors = []
arousal_errors = []

with torch.no_grad():
    for batch in test_loader:
        if batch is None:
            continue

        mel = batch["mel"].to(DEVICE)
        mel_lens = batch["mel_lens"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        tok_lens = batch["tok_lens"].to(DEVICE)
        genre = batch["genre"].to(DEVICE)

        outputs = model(mel, mel_lens, tokens, tok_lens)

        preds = torch.argmax(outputs["genre"], dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(genre.cpu().numpy())

        # Regression errors
        valence_errors.extend(
            (outputs["valence"] - batch["valence"].to(DEVICE)).abs().cpu().numpy()
        )
        arousal_errors.extend(
            (outputs["arousal"] - batch["arousal"].to(DEVICE)).abs().cpu().numpy()
        )

# =========================
# Metrics
# =========================
accuracy = accuracy_score(all_targets, all_preds)

print("\n🎯 TEST RESULTS")
print("--------------------------------------------------")
print(f"🎵 Genre Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(
    all_targets,
    all_preds,
    target_names=MAIN_GENRES,
    digits=4
))

print("📉 Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

print("\n🎼 Emotion Regression Error:")
print(f"Valence MAE: {np.mean(valence_errors):.4f}")
print(f"Arousal MAE: {np.mean(arousal_errors):.4f}")

print("\n✅ Testing complete!")
