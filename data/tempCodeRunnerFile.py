import os
import pandas as pd

# Paths
audio_dir = "data/audio"
lyrics_dir = "data/lyrics"
labels_file = "data/labels.csv"

# List files
audio_files = [f.replace(".mp3", "") for f in os.listdir(audio_dir) if f.endswith(".mp3")]
lyrics_files = [f.replace(".txt", "") for f in os.listdir(lyrics_dir) if f.endswith(".txt")]

# Read labels
labels_df = pd.read_csv(labels_file, encoding='utf-8-sig')
label_songs = labels_df['song_id'].tolist()  # adjust column name if different

# All songs combined
all_songs = set(audio_files) | set(lyrics_files) | set(label_songs)

# Prepare a table
missing_data = []

for song in sorted(all_songs):
    missing = []
    if song not in audio_files:
        missing.append("Audio")
    if song not in lyrics_files:
        missing.append("Lyrics")
    if song not in label_songs:
        missing.append("Label")
    if missing:  # only include songs with missing data
        missing_data.append([song, ", ".join(missing)])

# Convert to DataFrame
missing_df = pd.DataFrame(missing_data, columns=["Song", "Missing"])

# Print counts
print(f"Audio files count: {len(audio_files)}")
print(f"Lyrics files count: {len(lyrics_files)}")
print(f"Labels count: {len(label_songs)}")
print("\n⚠️ Songs with missing sources:\n")
print(missing_df.to_string(index=False))
