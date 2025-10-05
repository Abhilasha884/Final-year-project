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

# Find songs with missing sources
missing_in_audio = (lyrics_set | labels_set) - audio_set
missing_in_lyrics = (audio_set | labels_set) - lyrics_set
missing_in_labels = (audio_set | lyrics_set) - labels_set

# Combine all missing songs
all_missing_songs = missing_in_audio | missing_in_lyrics | missing_in_labels

print(f"Total songs to be deleted: {len(all_missing_songs)}\n")

# Delete files and remove labels
for song in sorted(all_missing_songs):
    audio_path = os.path.join(audio_dir, song + ".mp3")
    lyrics_path = os.path.join(lyrics_dir, song + ".txt")
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted audio: {song}.mp3")
    if os.path.exists(lyrics_path):
        os.remove(lyrics_path)
        print(f"Deleted lyrics: {song}.txt")
    
    if song in labels_set:
        labels_df = labels_df[labels_df['song_id'] != song]
        print(f"Deleted label entry: {song}")

# Save the updated labels CSV
labels_df.to_csv(labels_file, index=False)
print("\nUpdated labels.csv saved!")
