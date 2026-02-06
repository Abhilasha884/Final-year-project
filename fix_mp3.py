import os

AUDIO_DIR = "data/audio"

print("Scanning:", os.path.abspath(AUDIO_DIR))

for filename in os.listdir(AUDIO_DIR):
    old_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.isfile(old_path):
        continue

    # ✅ Skip files that already end with .mp3.mp3
    if filename.lower().endswith(".mp3.mp3"):
        print(f"✔ Already correct: {filename}")
        continue

    # Remove all extensions
    base = filename
    while "." in base:
        base = os.path.splitext(base)[0]

    new_filename = base + ".mp3.mp3"
    new_path = os.path.join(AUDIO_DIR, new_filename)

    # Avoid overwriting
    if os.path.exists(new_path):
        print(f"⚠️ Target exists, skipping: {new_filename}")
        continue

    os.rename(old_path, new_path)
    print(f"✅ Renamed: {filename} → {new_filename}")

print("🎵 Filename normalization complete (.mp3.mp3 enforced)")
