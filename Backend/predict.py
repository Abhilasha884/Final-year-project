import os
import tempfile
import numpy as np
import torch
import librosa
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import ModelConfig, MultiTaskMultimodalLSTM
# from train import simple_tokenizer

# -------------------- CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_GENRE = {
    0: "Pop",
    1: "Rock",
    2: "Hip-Hop",
    3: "R&B",
    4: "Electronic",
    5: "Jazz",
    6: "Classical",
    7: "Country"
}

# -------------------- LOAD MODEL --------------------
# cfg = ModelConfig(
#     n_mels=80,
#     n_genre=8,
#     vocab_size=256,
#     pad_idx=0,
#     hidden_size=64,
#     num_layers=2
# )
cfg = ModelConfig(
    n_mels=123,
    n_genre=8,
    vocab_size=256,
    pad_idx=0,
    hidden_size=128,
    num_layers=2
)

model = MultiTaskMultimodalLSTM(cfg).to(DEVICE)
model.load_state_dict(torch.load("Save_model/best_model.pth", map_location=DEVICE))
model.eval()

def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]

# -------------------- LOAD DATASET (FOR RECOMMENDATION) --------------------
CSV_FILE = "../data/labels.csv"
df = pd.read_csv(CSV_FILE)

# keep only valid main genres
df = df[df["genre"].isin(IDX_TO_GENRE.values())]

# -------------------- AUDIO UTILS --------------------
def extract_mel(path):
    wav, sr = librosa.load(path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_mels=123,
        n_fft=2048,
        hop_length=512
    )

    mel = librosa.power_to_db(mel)

    mel = torch.tensor(mel).transpose(0, 1).unsqueeze(0).float()
    mel_len = torch.tensor([mel.shape[1]])

    return mel, mel_len


def process_lyrics(text):
    tokens = simple_tokenizer(text)
    tokens = torch.tensor(tokens).unsqueeze(0)
    tok_len = torch.tensor([tokens.shape[1]])
    return tokens, tok_len


# -------------------- RECOMMENDATION LOGIC --------------------
def recommend_songs(pred_genre, valence, arousal, top_k=10):
    genre_df = df[df["genre"] == pred_genre]

    if genre_df.empty:
        return []

    genre_df = genre_df.copy()

    genre_df["emotion_distance"] = np.sqrt(
        (genre_df["valence"] - valence) ** 2 +
        (genre_df["arousal"] - arousal) ** 2
    )

    top = genre_df.sort_values("emotion_distance").head(top_k)

    return top["song_id"].tolist()


# -------------------- FLASK --------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        lyrics = request.form.get("lyrics", "")

        mel = mel_len = None
        if "file" in request.files:
            f = request.files["file"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                f.save(tmp.name)
                mel, mel_len = extract_mel(tmp.name)
            os.unlink(tmp.name)

        tokens, tok_len = process_lyrics(lyrics)

        mel = mel.to(DEVICE) if mel is not None else None
        mel_len = mel_len.to(DEVICE) if mel_len is not None else None
        tokens = tokens.to(DEVICE)
        tok_len = tok_len.to(DEVICE)

        with torch.no_grad():
            outputs = model(mel, mel_len, tokens, tok_len)

        valence = float(outputs["valence"].cpu().item())
        arousal = float(outputs["arousal"].cpu().item())

        genre_idx = int(torch.argmax(outputs["genre"], dim=-1).item())
        genre = IDX_TO_GENRE.get(genre_idx, "Unknown")

        recommendations = recommend_songs(genre, valence, arousal)

        return jsonify({
            "predicted_genre": genre,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "recommendations": recommendations
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": "Error analyzing song"}), 500


if __name__ == "__main__":
    app.run(debug=True)
