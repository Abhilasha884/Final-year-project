# # import os
# # import re
# # import torch
# # import librosa
# # import numpy as np
# # import pandas as pd

# # class MusicDataset(torch.utils.data.Dataset):
# #     def __init__(self, csv_file, audio_dir, lyrics_dir, genre_map=None, tokenizer=None, max_len=130):
# #         self.data = pd.read_csv(csv_file, encoding="utf-8")
# #         self.audio_dir = audio_dir
# #         self.lyrics_dir = lyrics_dir
# #         self.genre_map = genre_map
# #         self.tokenizer = tokenizer
# #         self.max_len = max_len

# #         # --------------------------
# #         # Normalizer for base names (no extension)
# #         # --------------------------
# #         def clean_name(name):
# #             name = str(name).lower().strip()
# #             name = name.strip("'").strip('"')
           
# #             # name = re.sub(r"[^a-z0-9_]", "_", name)
# #             # name = re.sub(r"_+", "_", name)
# #             return name.strip("_")

# #         self.clean_name = clean_name

# #         # --------------------------
# #         # Map audio files: key = normalized base name (WITHOUT extension), value = full path
# #         # --------------------------
# #         if not os.path.isdir(audio_dir):
# #             raise ValueError(f"audio_dir not found: {audio_dir}")

# #         self.audio_files = {}
# #         for f in os.listdir(audio_dir):
# #             if f.lower().endswith(".mp3"):
# #                 base = os.path.splitext(f)[0]
# #                 key = self.clean_name(base)
# #                 self.audio_files[key] = os.path.join(audio_dir, f)

# #         print(f"Found {len(self.audio_files)} audio files. Sample keys: {list(self.audio_files.keys())[:20]}")

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         row = self.data.iloc[idx]

       
# #         # base_from_csv = os.path.splitext(str(row["song_id"]))[0] + ".mp3"
# #         # song_key = self.clean_name(os.path.splitext(base_from_csv)[0])
# #         base_from_csv = os.path.splitext(str(row["song_id"]))[0]
# #         song_key = self.clean_name(base_from_csv)


# #         if song_key not in self.audio_files:
# #             raise FileNotFoundError(f"❌ Missing audio for: {song_key}")

# #         audio_path = self.audio_files[song_key]

# #         # --------------------------
# #         # Load audio -> mel spectrogram
# #         # --------------------------
# #         wav, sr = librosa.load(audio_path, sr=22050, mono=True)
# #         mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
# #         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# #         # Pad/trim to max_len (time frames)
# #         if mel_spec_db.shape[1] < self.max_len:
# #             pad_width = self.max_len - mel_spec_db.shape[1]
# #             mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode="constant")
# #         else:
# #             mel_spec_db = mel_spec_db[:, :self.max_len]

# #         mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).transpose(0, 1)

# #         # --------------------------
# #         # Lyrics
# #         # --------------------------
# #         lyrics_filename = base_from_csv.replace(".mp3", ".txt")
# #         lyrics_path = os.path.join(self.lyrics_dir, lyrics_filename)
# #         if os.path.exists(lyrics_path):
# #             with open(lyrics_path, "r", encoding="utf-8") as f:
# #                 text = f.read()
# #         else:
# #             text = ""

# #         tokens = torch.tensor(self.tokenizer(text), dtype=torch.long) if self.tokenizer else torch.tensor([0])
# #         if len(tokens) == 0:
# #             tokens = torch.tensor([0], dtype=torch.long)

# #         # --------------------------
# #         # Labels
# #         # --------------------------
# #         genre = None
# #         if self.genre_map and "genre" in row and pd.notna(row["genre"]):
# #             genre = torch.tensor(self.genre_map[row["genre"]], dtype=torch.long)

# #         valence = torch.tensor(row["valence"], dtype=torch.float32) if "valence" in row else torch.tensor(0.0)
# #         arousal = torch.tensor(row["arousal"], dtype=torch.float32) if "arousal" in row else torch.tensor(0.0)

# #         sample = {
# #             "mel": mel_tensor,
# #             "mel_len": mel_tensor.shape[0],
# #             "tokens": tokens,
# #             "tok_len": len(tokens),
# #             "valence": valence,
# #             "arousal": arousal,
# #         }
# #         if genre is not None:
# #             sample["genre"] = genre

# #         return sample

# # import os
# # import re
# # import torch
# # import librosa
# # import numpy as np
# # import pandas as pd
# # from difflib import get_close_matches
# # from genre_mapper import map_to_main_genre


# # class MusicDataset(torch.utils.data.Dataset):
# #     def __init__(
# #         self,
# #         csv_file,
# #         audio_dir,
# #         lyrics_dir,
# #         genre_map=None,
# #         tokenizer=None,
# #         max_len=130
# #     ):
# #         self.data = pd.read_csv(csv_file, encoding="utf-8")
# #         self.audio_dir = audio_dir
# #         self.lyrics_dir = lyrics_dir
# #         self.genre_map = genre_map
# #         self.tokenizer = tokenizer
# #         self.max_len = max_len

# #         def clean_name(name):
# #             name = str(name).lower().strip()
# #             name = name.strip("'").strip('"')
# #             name = name.replace("’", "'").replace("‘", "'")
# #             name = name.replace("–", "-").replace("—", "-")
# #             name = re.sub(r"\s+", "_", name)
# #             name = re.sub(r"[^a-z0-9._-]+", "_", name)
# #             name = re.sub(r"_+", "_", name)
# #             return name.strip("_")

# #         self.clean_name = clean_name

# #         if not os.path.isdir(audio_dir):
# #             raise ValueError(f"audio_dir not found: {audio_dir}")

# #         self.audio_files = {}
# #         for f in os.listdir(audio_dir):
# #             if f.lower().endswith(".mp3"):
# #                 key = self.clean_name(os.path.splitext(f)[0])
# #                 self.audio_files[key] = os.path.join(audio_dir, f)

# #         print(f"✅ Found {len(self.audio_files)} audio files.")
# #         print(f"🔑 Sample keys: {list(self.audio_files.keys())[:20]}")

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         row = self.data.iloc[idx]

# #         # ---------- AUDIO ----------
# #         base_from_csv = str(row["song_id"]).strip().lower()
# #         song_key = self.clean_name(os.path.splitext(base_from_csv)[0])

# #         if song_key not in self.audio_files:
# #             match = get_close_matches(song_key, self.audio_files.keys(), n=1, cutoff=0.6)
# #             if match:
# #                 song_key = match[0]
# #             else:
# #                 raise FileNotFoundError(f"❌ Missing audio for {song_key}")

# #         wav, sr = librosa.load(self.audio_files[song_key], sr=22050, mono=True)
# #         mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
# #         mel = librosa.power_to_db(mel, ref=np.max)

# #         if mel.shape[1] < self.max_len:
# #             mel = np.pad(mel, ((0, 0), (0, self.max_len - mel.shape[1])))
# #         else:
# #             mel = mel[:, :self.max_len]

# #         mel_tensor = torch.tensor(mel, dtype=torch.float32).transpose(0, 1)

# #         # ---------- LYRICS ----------
# #         lyrics_file = os.path.join(self.lyrics_dir, os.path.splitext(base_from_csv)[0] + ".txt")
# #         if os.path.exists(lyrics_file):
# #             with open(lyrics_file, "r", encoding="utf-8", errors="ignore") as f:
# #                 text = f.read()
# #         else:
# #             text = ""

# #         tokens = (
# #             torch.tensor(self.tokenizer(text), dtype=torch.long)
# #             if self.tokenizer
# #             else torch.tensor([0])
# #         )

# #         if len(tokens) == 0:
# #             tokens = torch.tensor([0], dtype=torch.long)

# #         # ---------- GENRE (MAIN GENRE ONLY) ----------
# #         genre = None
# #         if self.genre_map is not None and "genre" in row and pd.notna(row["genre"]):
# #             raw_genre = str(row["genre"]).strip()
# #             main_genre = map_to_main_genre(raw_genre)

# #             if main_genre and main_genre in self.genre_map:
# #                 genre = torch.tensor(self.genre_map[main_genre], dtype=torch.long)

# #         # ---------- EMOTIONS ----------
# #         valence = torch.tensor(row["valence"], dtype=torch.float32)
# #         arousal = torch.tensor(row["arousal"], dtype=torch.float32)

# #         sample = {
# #             "mel": mel_tensor,
# #             "mel_len": mel_tensor.shape[0],
# #             "tokens": tokens,
# #             "tok_len": len(tokens),
# #             "valence": valence,
# #             "arousal": arousal,
# #             "song_id": row["song_id"]
# #         }

# #         if genre is not None:
# #             sample["genre"] = genre

# #         return sample


# import os
# import re
# import torch
# import librosa
# import numpy as np
# import pandas as pd
# from difflib import get_close_matches


# class MusicDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         csv_file,
#         audio_dir,
#         lyrics_dir,
#         genre_map,
#         tokenizer=None,
#         max_len=2600  # ~60 seconds
#     ):
#         self.data = pd.read_csv(csv_file, encoding="utf-8")
#         self.audio_dir = audio_dir
#         self.lyrics_dir = lyrics_dir
#         self.genre_map = genre_map
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#         def clean_name(name):
#             name = str(name).lower().strip()
#             name = re.sub(r"\s+", "_", name)
#             name = re.sub(r"[^a-z0-9._-]+", "_", name)
#             return name.strip("_")

#         self.clean_name = clean_name

#         # Map audio files
#         self.audio_files = {
#             self.clean_name(os.path.splitext(f)[0]): os.path.join(audio_dir, f)
#             for f in os.listdir(audio_dir)
#             if f.lower().endswith(".mp3")
#         }

#         print(f"✅ Audio files found: {len(self.audio_files)}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]

#         # ================= AUDIO =================
#         song_key = self.clean_name(row["song_id"])

#         if song_key not in self.audio_files:
#             match = get_close_matches(song_key, self.audio_files.keys(), n=1, cutoff=0.6)
#             if not match:
#                 return None
#             song_key = match[0]

#         wav, sr = librosa.load(
#             self.audio_files[song_key],
#             sr=22050,
#             mono=True
#         )

#         mel = librosa.feature.melspectrogram(
#             y=wav,
#             sr=sr,
#             n_mels=80,
#             n_fft=2048,
#             hop_length=512
#         )

#         # 🔥 Log-mel
#         mel = librosa.power_to_db(mel)

#         # 🔥 Mean-Variance Normalization (CRITICAL)
#         mel = (mel - mel.mean()) / (mel.std() + 1e-6)

#         # Pad / trim to 60 seconds
#         mel = mel[:, :self.max_len]
#         if mel.shape[1] < self.max_len:
#             mel = np.pad(
#                 mel,
#                 ((0, 0), (0, self.max_len - mel.shape[1])),
#                 mode="constant"
#             )

#         mel = torch.tensor(mel, dtype=torch.float32).transpose(0, 1)

#         # ================= LYRICS =================
#         lyrics_path = os.path.join(self.lyrics_dir, f"{row['song_id']}.txt")
#         if os.path.exists(lyrics_path):
#             with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
#                 text = f.read()
#         else:
#             text = ""

#         tokens = torch.tensor(self.tokenizer(text), dtype=torch.long) if self.tokenizer else torch.tensor([0])
#         if len(tokens) == 0:
#             tokens = torch.tensor([0])

#         # ================= GENRE (FROM FINAL CSV) =================
#         if pd.isna(row["main_genre"]) or row["main_genre"] not in self.genre_map:
#             return None

#         genre = torch.tensor(self.genre_map[row["main_genre"]], dtype=torch.long)

#         # ================= EMOTION =================
#         valence = torch.tensor(row["valence"], dtype=torch.float32)
#         arousal = torch.tensor(row["arousal"], dtype=torch.float32)

#         return {
#             "mel": mel,
#             "mel_len": mel.shape[0],
#             "tokens": tokens,
#             "tok_len": len(tokens),
#             "genre": genre,
#             "valence": valence,
#             "arousal": arousal,
#         }



#dataset.py
import os
import re
import torch
import numpy as np
import pandas as pd
from difflib import get_close_matches


class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        lyrics_dir,
        genre_map,
        tokenizer=None,
        max_len=2600
    ):
        self.data = pd.read_csv(csv_file, encoding="utf-8")
        self.audio_dir = audio_dir
        self.lyrics_dir = lyrics_dir
        self.genre_map = genre_map
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 🔹 Feature directory (same level as audio)
        self.feature_dir = os.path.join(os.path.dirname(audio_dir), "features")

        def clean_name(name):
            name = str(name).lower().strip()
            name = re.sub(r"\s+", "_", name)
            name = re.sub(r"[^a-z0-9._-]+", "_", name)
            return name.strip("_")

        self.clean_name = clean_name

        # Keep audio map ONLY for matching names
        self.audio_files = {
            self.clean_name(os.path.splitext(f)[0]): f
            for f in os.listdir(audio_dir)
            if f.lower().endswith(".mp3")
        }

        print(f"✅ Audio files found: {len(self.audio_files)}")
        print(f"✅ Feature folder: {self.feature_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ================= SONG MATCHING =================
        song_key = self.clean_name(row["song_id"])

        if song_key not in self.audio_files:
            match = get_close_matches(song_key, self.audio_files.keys(), n=1, cutoff=0.6)
            if not match:
                return None
            song_key = match[0]

        # ================= LOAD FEATURES =================
        feature_path = os.path.join(self.feature_dir, song_key + ".npy")

        if not os.path.exists(feature_path):
            return None

        features = np.load(feature_path)

        # shape: (123 , 2600) → convert to (2600 , 123)
        features = torch.tensor(features, dtype=torch.float32).transpose(0, 1)

        # ================= LYRICS =================
        lyrics_path = os.path.join(self.lyrics_dir, f"{row['song_id']}.txt")

        if os.path.exists(lyrics_path):
            with open(lyrics_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            text = ""

        tokens = torch.tensor(self.tokenizer(text), dtype=torch.long) if self.tokenizer else torch.tensor([0])

        if len(tokens) == 0:
            tokens = torch.tensor([0])

        # ================= GENRE =================
        if pd.isna(row["main_genre"]) or row["main_genre"] not in self.genre_map:
            return None

        genre = torch.tensor(self.genre_map[row["main_genre"]], dtype=torch.long)

        # ================= EMOTION =================
        valence = torch.tensor(row["valence"], dtype=torch.float32)
        arousal = torch.tensor(row["arousal"], dtype=torch.float32)

        return {
            "mel": features,
            "mel_len": features.shape[0],
            "tokens": tokens,
            "tok_len": len(tokens),
            "genre": genre,
            "valence": valence,
            "arousal": arousal,
        }