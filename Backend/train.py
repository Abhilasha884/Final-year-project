import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from dataset import MusicDataset
from model import ModelConfig, MultiTaskMultimodalLSTM, compute_multitask_loss
from torch.optim.lr_scheduler import StepLR

# --------------------------
# Collate function (padding)
# --------------------------
def collate_fn(batch):
    # filter out None samples
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    # Pad mel spectrograms
    mel_lens = [item["mel_len"] for item in batch]
    mel_padded = pad_sequence([item["mel"] for item in batch], batch_first=True)

    # Pad tokens
    tok_lens = [item["tok_len"] for item in batch]
    tokens_padded = pad_sequence([item["tokens"] for item in batch],
                                 batch_first=True, padding_value=0)

    # Collect labels
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
        "valence": valences,
        "arousal": arousals,
    }
    if genres is not None:
        result["genre"] = genres

    return result

# --------------------------
# Paths
# --------------------------
CSV_FILE = "../data/labels.csv"
AUDIO_DIR = "../data/audio"
LYRICS_DIR = "../data/lyrics"

# --------------------------
# Load labels
# --------------------------
df = pd.read_csv(CSV_FILE, encoding="utf-8")

# Genre map (optional)
have_genre = "genre" in df.columns and df["genre"].notna().any()
genre_map = {g: i for i, g in enumerate(sorted(df["genre"].dropna().unique()))} if have_genre else None
print("Genres:", genre_map if genre_map else None)

# --------------------------
# Dummy tokenizer
# --------------------------
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]

# --------------------------
# Dataset & Dataloader
# --------------------------
train_dataset = MusicDataset(
    CSV_FILE, AUDIO_DIR, LYRICS_DIR,
    genre_map=genre_map,
    tokenizer=simple_tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# --------------------------
# Model setup
# --------------------------
vocab_size = 256
pad_idx = 0

cfg = ModelConfig(
    n_mels=80,
    n_genre=len(genre_map) if have_genre else None,
    vocab_size=vocab_size,
    pad_idx=pad_idx,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskMultimodalLSTM(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# --------------------------
# Training loop
# --------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        if batch is None:
            continue

        mel_spec = batch["mel"].to(device)
        mel_lens = batch["mel_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)

        optimizer.zero_grad()
        outputs = model(mel_spec, mel_lens, tokens, tok_lens)

        # Targets (no emotion)
        targets = {
            "valence": batch["valence"].to(device),
            "arousal": batch["arousal"].to(device),
        }
        if "genre" in batch:
            targets["genre"] = batch["genre"].to(device)

        loss, loss_items = compute_multitask_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
