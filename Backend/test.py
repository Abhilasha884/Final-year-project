# test.py

import os
import re
import torch
import librosa
import numpy as np
import pandas as pd
from difflib import get_close_matches
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer
from torch.cuda.amp import autocast

from data.model import ModelConfig, MultiTaskMultimodalLSTM
from data.genre_mapper import MAIN_GENRES


# =========================
# CONFIG
# =========================
BASE_PATH = "data"

CSV_FILE = os.path.join(BASE_PATH, "labels_test.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "audio")
LYRICS_DIR = os.path.join(BASE_PATH, "lyrics")

MODEL_PATH = "Save_model/best_model.pth"

BATCH_SIZE = 8
MAX_AUDIO_LEN = 2600
TEXT_MAX_LEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", DEVICE)


# =========================
# TOKENIZER (BERT)
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================
# AUDIO FEATURE EXTRACTION
# (MUST MATCH TRAINING)
# =========================
def extract_features(audio_path, max_len=2600):
    try:
        wav, sr = librosa.load(audio_path, sr=22050, mono=True)

        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
        mel = librosa.power_to_db(mel, ref=np.max)

        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=wav, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(wav)
        spec_contrast = librosa.feature.spectral_contrast(y=wav, sr=sr)

        tempo, _ = librosa.beat.beat_track(y=wav, sr=sr)
        tempo_feature = np.full((1, mel.shape[1]), float(tempo))

        features = np.concatenate([
            mel, mfcc, chroma,
            spec_centroid, spec_bw,
            zcr, spec_contrast,
            tempo_feature
        ], axis=0)

        # MUST be 123 (matches model input)
        if features.shape[0] != 123:
            return None

        # SAME NORMALIZATION AS TRAINING
        features = (features - features.mean()) / (features.std() + 1e-6)

        # PAD / TRIM
        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)))

        return torch.tensor(features.T, dtype=torch.float32)

    except Exception as e:
        print("Feature error:", e)
        return None


# =========================
# DATASET
# =========================
class TestDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(CSV_FILE)

        def clean_name(name):
            name = str(name).lower().strip()
            name = re.sub(r"\s+", "_", name)
            name = re.sub(r"[^a-z0-9._-]+", "_", name)
            return name.strip("_")

        self.clean_name = clean_name

        self.audio_files = {
            self.clean_name(os.path.splitext(f)[0]): f
            for f in os.listdir(AUDIO_DIR)
            if f.endswith(".mp3")
        }

        print(f"✅ Audio files: {len(self.audio_files)}")
        print(f"✅ Test samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]

            song_id = str(row["song_id"])
            key = self.clean_name(song_id)

            if key not in self.audio_files:
                match = get_close_matches(key, self.audio_files.keys(), n=1, cutoff=0.6)
                if not match:
                    return None
                key = match[0]

            audio_path = os.path.join(AUDIO_DIR, self.audio_files[key])

            mel = extract_features(audio_path)
            if mel is None:
                return None

            # ================= TEXT =================
            lyrics_path = os.path.join(LYRICS_DIR, f"{song_id}.txt")
            text = ""

            if os.path.exists(lyrics_path):
                with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            # CLEAN TEXT (same as training)
            text = text.lower()
            text = re.sub(r"\[.*?\]", " ", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=TEXT_MAX_LEN,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            genre = torch.tensor(MAIN_GENRES.index(row["main_genre"]))
            valence = torch.tensor(float(row["valence"]))
            arousal = torch.tensor(float(row["arousal"]))

            return {
                "mel": mel,
                "mel_len": mel.shape[0],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "genre": genre,
                "valence": valence,
                "arousal": arousal
            }

        except Exception as e:
            print("Sample error:", e)
            return None


# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    return {
        "mel": pad_sequence([x["mel"] for x in batch], batch_first=True),
        "mel_lens": torch.tensor([x["mel_len"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "genre": torch.stack([x["genre"] for x in batch]),
        "valence": torch.stack([x["valence"] for x in batch]),
        "arousal": torch.stack([x["arousal"] for x in batch]),
    }


# =========================
# LOAD MODEL
# =========================
model = MultiTaskMultimodalLSTM(
    ModelConfig(n_mels=123, n_genre=len(MAIN_GENRES))
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print(" Model loaded")


# =========================
# TEST LOOP
# =========================
dataset = TestDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

all_preds, all_targets = [], []
valence_err, arousal_err = [], []

with torch.no_grad():
    for batch in loader:
        if batch is None:
            continue

        with autocast():
            outputs = model(
                batch["mel"].to(DEVICE),
                batch["mel_lens"].to(DEVICE),
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE)
            )

        preds = torch.argmax(outputs["genre"], dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch["genre"].numpy())

        valence_err.extend(
            torch.abs(outputs["valence"] - batch["valence"].to(DEVICE)).cpu().numpy()
        )
        arousal_err.extend(
            torch.abs(outputs["arousal"] - batch["arousal"].to(DEVICE)).cpu().numpy()
        )


# =========================
# RESULTS
# =========================
print("\n RESULTS")
print("Accuracy:", accuracy_score(all_targets, all_preds))

labels_present = sorted(set(all_targets))

print("\n Classification Report:")
print(classification_report(
    all_targets,
    all_preds,
    labels=labels_present,
    target_names=[MAIN_GENRES[i] for i in labels_present]
))

print("\n Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))

print("\n Emotion Metrics")
print("Valence MAE:", np.mean(valence_err))
print("Arousal MAE:", np.mean(arousal_err))

print("\n Done")
