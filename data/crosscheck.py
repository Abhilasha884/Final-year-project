# import os
# import re
# import pandas as pd
# from collections import Counter

# # Paths
# audio_dir = "data/audio"
# lyrics_dir = "data/lyrics"
# labels_file = "data/labels.csv"

# # --- Improved normalization helper ---
# def normalize(name):
#     """Clean and standardize song names for matching."""
#     name = name.lower().strip()
#     name = name.replace(".mp3", "").replace(".txt", "")
#     # Remove language prefixes like english_, hindi_, etc.
#     name = re.sub(r"^(english|hindi|spanish|korean|french|japanese)_", "", name)
#     # Remove common suffixes like _trimmed, _audio, _official, _lyrics, etc.
#     name = re.sub(r"(_trimmed|_audio|_official|_lyrics|_video|_song)$", "", name)
#     # Replace underscores and hyphens with spaces
#     name = name.replace("_", " ").replace("-", " ")
#     # Remove extra spaces
#     name = re.sub(r"\s+", " ", name).strip()
#     return name

# # --- List and normalize file names ---
# audio_files = [normalize(f) for f in os.listdir(audio_dir) if f.endswith(".mp3")]
# lyrics_files = [normalize(f) for f in os.listdir(lyrics_dir) if f.endswith(".txt")]

# # --- Read labels and normalize ---
# labels_df = pd.read_csv(labels_file, encoding='utf-8-sig')
# label_songs = [normalize(str(s)) for s in labels_df['song_id'].tolist()]  # adjust column name if needed

# # --- DEBUG: print samples ---
# print("=== DEBUG: SAMPLE ENTRIES ===")
# print("Audio files (normalized):", audio_files[:10])
# print("Lyrics files (normalized):", lyrics_files[:10])
# print("Label songs (normalized):", label_songs[:10])
# print("==============================\n")

# # --- Combine all unique song names ---
# all_songs = set(audio_files) | set(lyrics_files) | set(label_songs)

# # --- Detect missing data ---
# missing_data = []
# for song in sorted(all_songs):
#     missing = []
#     if song not in audio_files:
#         missing.append("Audio")
#     if song not in lyrics_files:
#         missing.append("Lyrics")
#     if song not in label_songs:
#         missing.append("Label")
#     if missing:
#         missing_data.append([song, ", ".join(missing)])

# # --- Convert results to DataFrame ---
# missing_df = pd.DataFrame(missing_data, columns=["Song", "Missing"])

# # --- Print summary ---
# print(f"Audio files count: {len(audio_files)}")
# print(f"Lyrics files count: {len(lyrics_files)}")
# print(f"Labels count: {len(label_songs)}")
# print("\n⚠️ Songs with missing sources:\n")
# print(missing_df.to_string(index=False) if not missing_df.empty else "✅ No missing songs found.")

# # --- EXTRA CHECK: find audio-only songs ---
# extra_audio = set(audio_files) - (set(lyrics_files) | set(label_songs))
# if extra_audio:
#     print("\n🎧 Extra audio files (no lyrics/labels found):")
#     for song in sorted(extra_audio):
#         print("-", song)
# else:
#     print("\n✅ No extra standalone audio files found.")

# # --- DUPLICATE CHECK: detect duplicates after normalization ---
# dupes = [name for name, count in Counter(audio_files).items() if count > 1]
# if dupes:
#     print("\n🔁 Duplicate audio entries after normalization:")
#     for d in dupes:
#         print("-", d)
# else:
#     print("\n✅ No duplicate audio entries found.")


import os
import re
from collections import defaultdict

audio_dir = "data/audio"

def normalize(name):
    name = name.lower().strip()
    name = name.replace(".mp3", "")
    name = re.sub(r"^(english|hindi|spanish|korean|french|japanese)_", "", name)
    name = re.sub(r"(_trimmed|_audio|_official|_lyrics|_video|_song)$", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name

# Map normalized name -> original files
file_map = defaultdict(list)

for file in os.listdir(audio_dir):
    if file.endswith(".mp3"):
        norm = normalize(file)
        file_map[norm].append(file)

# Delete duplicates
deleted = []

for norm_name, files in file_map.items():
    if len(files) > 1:
        # keep first file, delete rest
        for dup in files[1:]:
            path = os.path.join(audio_dir, dup)
            os.remove(path)
            deleted.append(dup)

print("Deleted duplicate files:")
for d in deleted:
    print("-", d)

print(f"\nTotal duplicates removed: {len(deleted)}")
