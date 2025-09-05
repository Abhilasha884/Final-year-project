import os
import torch
import librosa
import numpy as np
import pandas as pd


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, lyrics_dir, genre_map, emotion_map, tokenizer=None, max_len=130):
        # Load CSV
        self.data = pd.read_csv(csv_file)

        # -------------------
        # Clean CSV song_id and lyrics_file
        # -------------------
        self.data['song_id'] = self.data['song_id'].astype(str).str.strip()
        self.data['song_id'] = self.data['song_id'].str.replace("\xa0", "", regex=False)
        self.data['song_id'] = self.data['song_id'].str.replace("\u200b", "", regex=False)

        if 'lyrics_file' in self.data.columns:
            self.data['lyrics_file'] = self.data['lyrics_file'].astype(str).str.strip()
            self.data['lyrics_file'] = self.data['lyrics_file'].str.replace("\xa0", "", regex=False)
            self.data['lyrics_file'] = self.data['lyrics_file'].str.replace("\u200b", "", regex=False)

        self.audio_dir = audio_dir
        self.lyrics_dir = lyrics_dir
        self.genre_map = genre_map
        self.emotion_map = emotion_map
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # -------------------
        # Normalize song_id
        # -------------------
        song_id = str(row["song_id"]).strip()
        song_id = song_id.replace("\xa0", "").replace("\u200b", "")

        if not song_id.lower().endswith(".mp3"):
            song_id += ".mp3"

        audio_path = os.path.join(self.audio_dir, song_id)

        # -------------------
        # Case-insensitive match with cleaned folder files
        # -------------------
        if not os.path.exists(audio_path):
            audio_files = [f.strip().replace("\xa0", "").replace("\u200b", "") for f in os.listdir(self.audio_dir)]
            match = [f for f in audio_files if f.lower() == song_id.lower()]
            if match:
                audio_path = os.path.join(self.audio_dir, match[0])
            else:
                raise FileNotFoundError(
                    f"❌ Missing audio file: {audio_path}\n"
                    f"Existing files in folder: {audio_files}"
                )

        # -------------------
        # Load audio features
        # -------------------
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

        # -------------------
        # Lyrics tokens
        # -------------------
        lyrics_file = str(row["lyrics_file"]).strip() if "lyrics_file" in row else ""
        lyrics_file = lyrics_file.replace("\xa0", "").replace("\u200b", "")
        lyrics_path = os.path.join(self.lyrics_dir, lyrics_file)

        if os.path.exists(lyrics_path):
            with open(lyrics_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""

        tokens = torch.tensor(self.tokenizer(text), dtype=torch.long) if self.tokenizer else torch.tensor([])

        # -------------------
        # Labels
        # -------------------
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
