import os
import pandas as pd

# Paths
audio_dir = "data/audio"
lyrics_dir = "data/lyrics"
labels_file = "data/labels.csv"

# --- Normalization helper ---
def normalize(name):
    """Clean and standardize song names for comparison."""
    name = name.lower().strip()
    name = name.replace(".mp3", "").replace(".txt", "")
    # remove common language prefixes like 'english_', 'hindi_', etc.
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0] in ["english", "hindi", "spanish", "korean", "french", "japanese"]:
            name = parts[1]
    return name

# --- List and normalize file names ---
audio_files = [normalize(f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]
lyrics_files = [normalize(f) for f in os.listdir(lyrics_dir) if f.endswith(".txt")]

# --- Read labels and normalize ---
labels_df = pd.read_csv(labels_file, encoding='utf-8-sig')
label_songs = [normalize(str(s)) for s in labels_df['song_id'].tolist()]  # adjust column name if needed

# --- Combine all songs ---
all_songs = set(audio_files) | set(lyrics_files) | set(label_songs)

# --- Check for missing data ---
missing_data = []
for song in sorted(all_songs):
    missing = []
    if song not in audio_files:
        missing.append("Audio")
    if song not in lyrics_files:
        missing.append("Lyrics")
    if song not in label_songs:
        missing.append("Label")
    if missing:
        missing_data.append([song, ", ".join(missing)])

# --- Results ---
missing_df = pd.DataFrame(missing_data, columns=["Song", "Missing"])

# --- Print summary ---
print(f"Audio files count: {len(audio_files)}")
print(f"Lyrics files count: {len(lyrics_files)}")
print(f"Labels count: {len(label_songs)}")
print("\n⚠️ Songs with missing sources:\n")
print(missing_df.to_string(index=False))
