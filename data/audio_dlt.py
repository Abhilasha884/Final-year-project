import os
import pandas as pd

audio_dir = "data/audio"
lyrics_dir = "data/lyrics"
labels_file = "data/labels.csv"

def normalize(name):
    name = name.lower().strip()
    name = name.replace(".mp3", "").replace(".txt", "")
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0] in ["english", "hindi", "spanish", "korean", "french", "japanese"]:
            name = parts[1]
    return name

# Get normalized sets
audio_files = {normalize(f): f for f in os.listdir(audio_dir) if f.endswith(".mp3")}
lyrics_files = {normalize(f) for f in os.listdir(lyrics_dir) if f.endswith(".txt")}

labels_df = pd.read_csv(labels_file, encoding="utf-8-sig")
label_songs = {normalize(str(s)) for s in labels_df["song_id"]}

# Find extra audios
extra_audios = []

for norm_name, original_file in audio_files.items():
    if norm_name not in lyrics_files or norm_name not in label_songs:
        extra_audios.append(original_file)

print(f"Extra audio files to delete: {len(extra_audios)}\n")

# Delete them
for file in extra_audios:
    path = os.path.join(audio_dir, file)
    os.remove(path)
    print(f"Deleted: {file}")

print("\n✅ Extra audio cleanup complete!")