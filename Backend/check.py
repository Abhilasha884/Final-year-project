import os
import pandas as pd

# Load CSV
df = pd.read_csv("data/labels.csv")

# Clean CSV values
csv_songs = [str(s).strip().replace("\xa0", "").replace("\u200b", "") for s in df['song_id']]
print("CSV song_ids:")
print([repr(s) for s in csv_songs[:20]])  # show first 20

# List audio files
audio_files = os.listdir("data/audio")
print("Audio files in folder:")
print([repr(f) for f in audio_files[:20]])  # show first 20