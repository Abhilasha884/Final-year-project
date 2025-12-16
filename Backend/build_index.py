import os
import json
import numpy as np
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

from dataset import MusicDataset
from model import ModelConfig, MultiTaskMultimodalLSTM
from train import simple_tokenizer, collate_fn
from genre_mapper import MAIN_GENRES, map_to_main_genre

# =========================
# PATHS
# =========================
CSV_FILE = "../data/labels.csv"
AUDIO_DIR = "../data/audio"
LYRICS_DIR = "../data/lyrics"

OUT_DIR = "backend_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "Save_model/best_model.pth"

# =========================
# GENRE SETUP (MAIN ONLY)
# =========================
print("🎵 Building index on MAIN genres only:")
GENRE_MAP = {g: i for i, g in enumerate(MAIN_GENRES)}
ID_TO_GENRE = {i: g for g, i in GENRE_MAP.items()}
print(GENRE_MAP)

# =========================
# DATASET
# =========================
dataset = MusicDataset(
    CSV_FILE,
    AUDIO_DIR,
    LYRICS_DIR,
    genre_map=GENRE_MAP,
    tokenizer=simple_tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# MODEL (MUST MATCH train.py)
# =========================
cfg = ModelConfig(
    n_mels=80,
    vocab_size=256,
    pad_idx=0,
    n_genre=len(GENRE_MAP),   # ✅ 8 MAIN genres
    hidden_size=64,
    num_layers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskMultimodalLSTM(cfg).to(device)

# =========================
# LOAD TRAINED MODEL
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)   # ✅ strict load (no mismatch now)
model.eval()

print("🔥 Loaded trained model for indexing")

# =========================
# BUILD EMBEDDINGS
# =========================
embeddings = []
metadata = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        if batch is None:
            continue

        mel = batch["mel"].to(device)
        mel_lens = batch["mel_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)

        outputs = model(mel, mel_lens, tokens, tok_lens)

        # ---- fused embedding ----
        fused = outputs.get("embedding")
        if fused is None:
            audio_repr = model.audio_encoder(mel, mel_lens)
            lyrics_repr = model.lyrics_encoder(tokens, tok_lens)
            fused = torch.cat([audio_repr, lyrics_repr], dim=-1)

        vec = fused[0].cpu().numpy()
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        embeddings.append(vec)

        # ---- metadata ----
        song_id = str(dataset.data.iloc[i]["song_id"])

        predicted_genre = None
        if outputs.get("genre") is not None:
            g_idx = int(torch.argmax(outputs["genre"], dim=-1).item())
            predicted_genre = ID_TO_GENRE.get(g_idx)

        metadata.append({
            "song_id": song_id,
            "predicted_genre": predicted_genre,
            "valence": float(outputs["valence"].cpu().item()),
            "arousal": float(outputs["arousal"].cpu().item())
        })

        if (i + 1) % 25 == 0:
            print(f"Processed {i+1}/{len(dataset)} songs")

# =========================
# SAVE ARTIFACTS
# =========================
embeddings = np.array(embeddings, dtype=np.float32)

np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

nn_model = NearestNeighbors(
    n_neighbors=50,
    metric="cosine",
    algorithm="auto"
)
nn_model.fit(embeddings)

joblib.dump(nn_model, os.path.join(OUT_DIR, "nn_model.joblib"))

print(" Indexing complete!")
print(" Saved artifacts to:", OUT_DIR)
