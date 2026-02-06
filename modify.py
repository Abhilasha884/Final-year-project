import os
import librosa
import soundfile as sf

# ---------------------------
# CONFIGURATION
# ---------------------------
audio_folder = "data/audio"
snippet_duration = 60  # in seconds
sr = 22050  # audio sampling rate

# ---------------------------
# PROCESS EACH AUDIO FILE
# ---------------------------
for file in os.listdir(audio_folder):
    if file.endswith(".mp3") or file.endswith(".wav"):
        audio_path = os.path.join(audio_folder, file)

        try:
            y, sr = librosa.load(audio_path, sr=sr)
            snippet_samples = snippet_duration * sr

            if len(y) > snippet_samples:
                y_snippet = y[:snippet_samples]  # take first 60 sec
            else:
                y_snippet = y  # if shorter than 60 sec, keep full

            sf.write(audio_path, y_snippet, sr)
            print(f"✅ Audio snippet saved: {file}")

        except Exception as e:
            print(f"❌ Error processing audio {file}: {e}")
