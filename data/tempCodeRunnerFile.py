import os
import pandas as pd

# Paths
audio_dir = "data/audio"
lyrics_dir = "data/lyrics"
labels_file = "data/labels.csv"

# List files
audio_files = [f.replace(".mp3", "") for f in os.listdir(audio_dir) if f.endswith(".mp3")]
lyrics_files = [f.replace(".txt", "") for f in os.listdir(lyrics_dir) if f.endswith(".txt")]
labels_df = pd.read_csv(labels_file)
label_songs = labels_df['song_id'].tolist()  # assuming 'song_id' is the column with song names

# Convert to sets
audio_set = set(audio_files)
lyrics_set = set(lyrics_files)
labels_set = set(label_songs)

# Find missing files
missing_in_audio = (lyrics_set | labels_set) - audio_set
missing_in_lyrics = (audio_set | labels_set) - lyrics_set
missing_in_labels = (audio_set | lyrics_set) - labels_set

# Print counts
print(f"Audio files count: {len(audio_set)}")
print(f"Lyrics files count: {len(lyrics_set)}")
print(f"Labels count: {len(labels_set)}")
print("\n⚠️ Missing files:")

if missing_in_audio:
    print("\nMissing in Audio:")
    for song in sorted(missing_in_audio):
        print(song)

if missing_in_lyrics:
    print("\nMissing in Lyrics:")
    for song in sorted(missing_in_lyrics):
        print(song)

if missing_in_labels:
    print("\nMissing in Labels:")
    for song in sorted(missing_in_labels):
        print(song)
