import librosa
import numpy as np
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import os

# ----------------------------
# Load sentiment model
# ----------------------------
sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512
)


# ----------------------------
# AUDIO → AROUSAL
# ----------------------------
def extract_audio_arousal(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    onset = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

    tempo_n = min(tempo / 200, 1.0)
    rms_n = min(rms / 0.1, 1.0)
    onset_n = min(onset / 5.0, 1.0)

    arousal = 0.4 * tempo_n + 0.4 * rms_n + 0.2 * onset_n
    return arousal


# ----------------------------
# LYRICS → VALENCE
# ----------------------------
def extract_lyrics_valence(text):
    result = sentiment(text)[0]

    if result["label"] == "LABEL_2":      # positive
        return 0.6 + 0.4 * result["score"]
    elif result["label"] == "LABEL_0":    # negative
        return 0.4 - 0.4 * result["score"]
    else:
        return 0.45


# ----------------------------
# FUSION
# ----------------------------
def fuse_va(curr_v, curr_a, lyric_v, audio_a):
    final_v = (
        0.50 * lyric_v +
        0.30 * curr_v +
        0.20 * audio_a
    )

    final_a = (
        0.60 * audio_a +
        0.40 * curr_a
    )
    

    final_v = np.clip(final_v, 0.05, 0.95)
    final_a = np.clip(final_a, 0.05, 0.95)

    return float(final_v), float(final_a)



# ----------------------------
# MAIN
# ----------------------------
df = pd.read_csv("data/labels.csv")


new_valence = []
new_arousal = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    sid = str(row["song_id"])

    audio_path = f"data/audio/{sid}.mp3"
    lyrics_path = f"data/lyrics/{sid}.txt"


    if not os.path.exists(audio_path) or not os.path.exists(lyrics_path):
        # fallback: keep original
        new_valence.append(row["valence"])
        new_arousal.append(row["arousal"])
        continue

    audio_a = extract_audio_arousal(audio_path)

    with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
        lyrics = f.read()

    lyric_v = extract_lyrics_valence(lyrics)

    v, a = fuse_va(
        row["valence"],
        row["arousal"],
        lyric_v,
        audio_a
    )

    new_valence.append(v)
    new_arousal.append(a)

df["valence"] = new_valence
df["arousal"] = new_arousal

df.to_csv("data/labels_song_specific_va.csv", index=False)

print("✅ Song-specific VA updated and saved!")
