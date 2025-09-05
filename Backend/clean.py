import os

audio_dir = "data/audio"

for f in os.listdir(audio_dir):
    new_name = f.strip().strip("'").strip('"')  # remove extra quotes
    new_path = os.path.join(audio_dir, new_name)
    old_path = os.path.join(audio_dir, f)
    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed: {f} -> {new_name}")
