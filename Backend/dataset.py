import os
import torch
import librosa
import numpy as np
import pandas as pd

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, lyrics_dir, genre_map, emotion_map, tokenizer=None, max_len=130):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.lyrics_dir = lyrics_dir
        self.genre_map = genre_map
        self.emotion_map = emotion_map
        self.tokenizer = tokenizer
        self.max_len = max_len

        # --------------------------
        # Build normalized audio file mapping
        # --------------------------
        cleaned_files = {}
        for f in os.listdir(audio_dir):
            # Normalize filename: lowercase, strip quotes, replace spaces with underscores
            name = f.lower().strip().strip("'").strip('"').replace(" ", "_")
            if name.endswith(".mp3"):
                name = name[:-4]  # remove .mp3 for key
            cleaned_files[name] = os.path.join(audio_dir, f)
        self.audio_files = cleaned_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        song_id = str(row["song_id"]).lower().strip().strip("'").strip('"').replace(" ", "_")

        # --------------------------
        # Match CSV song_id to actual audio file
        # --------------------------
        if song_id in self.audio_files:
            audio_path = self.audio_files[song_id]
        else:
            raise FileNotFoundError(
                f"❌ Missing audio file: {song_id}\n"
                f"Existing files in folder: {list(self.audio_files.keys())[:50]}..."
            )

        # --------------------------
        # Load audio features
        # --------------------------
        wav, sr = librosa.load(audio_path, sr=22050, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad/trim to fixed length
        if mel_spec_db.shape[1] < self.max_len:
            pad_width = self.max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]

        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).transpose(0, 1)  # (time, mel)

        # --------------------------
        # Lyrics tokens (match audio filename)
        # --------------------------
        lyrics_filename = song_id + ".txt"  # use same song_id for lyrics
        lyrics_path = os.path.join(self.lyrics_dir, lyrics_filename)
        if os.path.exists(lyrics_path):
            with open(lyrics_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""

        tokens = torch.tensor(self.tokenizer(text), dtype=torch.long) if self.tokenizer else torch.tensor([0])
        if len(tokens) == 0:
            tokens = torch.tensor([0], dtype=torch.long)  # add dummy token if lyrics empty



        # --------------------------
        # Labels
        # --------------------------
        emotion = torch.tensor(self.emotion_map[row["emotion_label"]], dtype=torch.long)

        genre = None
        if self.genre_map and "genre" in row and pd.notna(row["genre"]):
            genre = torch.tensor(self.genre_map[row["genre"]], dtype=torch.long)

        valence = torch.tensor(row["valence"], dtype=torch.float32) if "valence" in row else torch.tensor(0.0)
        arousal = torch.tensor(row["arousal"], dtype=torch.float32) if "arousal" in row else torch.tensor(0.0)

        sample = {
            "mel": mel_tensor,
            "mel_len": mel_tensor.shape[0],
            "tokens": tokens,
            "tok_len": len(tokens),
            "emotion": emotion,
            "valence": valence,
            "arousal": arousal,
        }
        if genre is not None:
            sample["genre"] = genre

        return sample
