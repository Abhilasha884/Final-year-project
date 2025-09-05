import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from dataset import MusicDataset
from model import ModelConfig, MultiTaskMultimodalLSTM, compute_multitask_loss


# --------------------------
# Collate function (padding)
# --------------------------
def collate_fn(batch):
    # Pad mel spectrograms
    mel_lens = [item["mel_len"] for item in batch]
    mel_padded = pad_sequence([item["mel"] for item in batch], batch_first=True)

    # Pad tokens
    tok_lens = [item["tok_len"] for item in batch]
    tokens_padded = pad_sequence([item["tokens"] for item in batch],
                                 batch_first=True, padding_value=0)

    # Collect labels
    emotions = torch.stack([item["emotion"] for item in batch])
    valences = torch.stack([item["valence"] for item in batch])
    arousals = torch.stack([item["arousal"] for item in batch])

    # Genre (optional)
    genres = None
    if "genre" in batch[0]:
        genres = torch.stack([item["genre"] for item in batch])

    result = {
        "mel": mel_padded,
        "mel_lens": torch.tensor(mel_lens),
        "tokens": tokens_padded,
        "tok_lens": torch.tensor(tok_lens),
        "emotion": emotions,
        "valence": valences,
        "arousal": arousals,
    }
    if genres is not None:
        result["genre"] = genres

    return result


# --------------------------
# Paths
# --------------------------
CSV_FILE = "Backend/data/labels.csv"
AUDIO_DIR = "Backend/data/audio"
LYRICS_DIR = "Backend/data/lyrics"

# --------------------------
# Load labels
# --------------------------
df = pd.read_csv(CSV_FILE)

# Build label maps
emotion_map = {e: i for i, e in enumerate(sorted(df["emotion_label"].unique()))}
have_genre = "genre" in df.columns and df["genre"].notna().any()
genre_map = {g: i for i, g in enumerate(sorted(df["genre"].dropna().unique()))} if have_genre else None

print("Genres:", genre_map if genre_map else None)
print("Emotions:", emotion_map)


# --------------------------
# Dummy tokenizer
# --------------------------
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]  # chars → integers


# --------------------------
# Dataset & Dataloader
# --------------------------
train_dataset = MusicDataset(CSV_FILE, AUDIO_DIR, LYRICS_DIR,
                             genre_map, emotion_map, tokenizer=simple_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


# --------------------------
# Model setup
# --------------------------
vocab_size = 256  # tokenizer maps chars into 0–255
pad_idx = 0

cfg = ModelConfig(
    n_mels=80,
    n_emotion=len(emotion_map),
    n_genre=len(genre_map) if have_genre else None,
    vocab_size=vocab_size,
    pad_idx=pad_idx,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskMultimodalLSTM(cfg).to(device)

# ✅ Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Add scheduler
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


# --------------------------
# Training loop (multimodal)
# --------------------------
EPOCHS = 20  # ✅ Train longer

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        mel_spec = batch["mel"].to(device)
        mel_lens = batch["mel_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)

        optimizer.zero_grad()

        # Forward multimodal model
        outputs = model(mel_spec, mel_lens, tokens, tok_lens)

        # Targets
        targets = {
            "emotion": batch["emotion"].to(device),
            "valence": batch["valence"].to(device),
            "arousal": batch["arousal"].to(device),
        }
        if "genre" in batch:
            targets["genre"] = batch["genre"].to(device)

        # Loss
        loss, loss_items = compute_multitask_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # ✅ update learning rate

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
