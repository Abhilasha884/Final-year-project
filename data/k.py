import os
import pandas as pd
import re

# Paths
audio_dir = "data/audio"
lyrics_dir = "data/lyrics"
labels_file = "data/labels.csv"
muse_file = "data/muse_v3.csv"
output_file = "missing_audio_tuples.txt"   # final tuple file

# --- Normalization helper ---
def normalize(name):
    name = name.lower().strip()
    name = name.replace(".mp3", "").replace(".txt", "")
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0] in ["english", "hindi", "spanish", "korean", "french", "japanese"]:
            name = parts[1]
    return name

def normalize_muse(text):
    text = str(text).lower()
    text = re.sub(r"[^\w]+", "_", text)
    return text.strip("_")

# --- List and normalize file names ---
audio_files = [normalize(f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]
lyrics_files = [normalize(f) for f in os.listdir(lyrics_dir) if f.endswith(".txt")]

# --- Read labels ---
labels_df = pd.read_csv(labels_file, encoding='utf-8-sig')
label_songs = [normalize(str(s)) for s in labels_df['song_id'].tolist()]

# --- Combine all songs ---
all_songs = set(audio_files) | set(lyrics_files) | set(label_songs)

# --- Find songs missing AUDIO ---
missing_audio = [song for song in all_songs if song not in audio_files]

print(f"Total songs missing audio: {len(missing_audio)}")

# --- Load muse dataset ---
muse_df = pd.read_csv(muse_file)
muse_df["norm_track"] = muse_df["track"].apply(normalize_muse)

# --- Create tuple list ---
tuple_lines = []

for song in missing_audio:
    match = muse_df[muse_df["norm_track"] == normalize_muse(song)]
    
    if not match.empty:
        row = match.iloc[0]
        tup = f'("{row["track"]}", "{row["artist"]}", {round(row["valence_tags"],2)}, {round(row["arousal_tags"],2)}),'
        tuple_lines.append(tup)

# --- Save tuples to file ---
with open(output_file, "w", encoding="utf-8") as f:
    for line in tuple_lines:
        f.write(line + "\n")

print(f"\n✅ Tuple file created: {output_file}")