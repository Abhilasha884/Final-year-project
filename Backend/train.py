

# # import os
# # import torch
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, random_split
# # from torch.nn.utils.rnn import pad_sequence
# # import pandas as pd
# # from torch.optim.lr_scheduler import StepLR

# # from dataset import MusicDataset
# # from model import ModelConfig, MultiTaskMultimodalLSTM
# # from genre_mapper import MAIN_GENRES

# # import numpy as np
# # from sklearn.utils.class_weight import compute_class_weight
# # import torch.nn.functional as F


# # # =========================
# # # Collate function
# # # =========================
# # def collate_fn(batch):
# #     batch = [x for x in batch if x is not None and "genre" in x]
# #     if len(batch) == 0:
# #         return None

# #     mel_lens = [item["mel_len"] for item in batch]
# #     mel_padded = pad_sequence([item["mel"] for item in batch], batch_first=True)

# #     tok_lens = [item["tok_len"] for item in batch]
# #     tokens_padded = pad_sequence(
# #         [item["tokens"] for item in batch],
# #         batch_first=True,
# #         padding_value=0
# #     )

# #     return {
# #         "mel": mel_padded,
# #         "mel_lens": torch.tensor(mel_lens),
# #         "tokens": tokens_padded,
# #         "tok_lens": torch.tensor(tok_lens),
# #         "valence": torch.stack([item["valence"] for item in batch]),
# #         "arousal": torch.stack([item["arousal"] for item in batch]),
# #         "genre": torch.stack([item["genre"] for item in batch]),
# #     }


# # # =========================
# # # Paths
# # # =========================
# # CSV_FILE = "../data/labels_mapped.csv"
# # AUDIO_DIR = "../data/audio"
# # LYRICS_DIR = "../data/lyrics"


# # # =========================
# # # Tokenizer
# # # =========================
# # def simple_tokenizer(text):
# #     return [ord(c) % 256 for c in text]


# # # =========================
# # # Load CSV & genre map
# # # =========================
# # df = pd.read_csv(CSV_FILE, encoding="utf-8")
# # have_genre = "genre" in df.columns and df["genre"].notna().any()

# # genre_map = {g: i for i, g in enumerate(MAIN_GENRES)} if have_genre else None

# # print("🎵 Training on MAIN genres only:")
# # print(genre_map)


# # # =========================
# # # Main
# # # =========================
# # if __name__ == "__main__":

# #     dataset = MusicDataset(
# #         CSV_FILE,
# #         AUDIO_DIR,
# #         LYRICS_DIR,
# #         genre_map=genre_map,
# #         tokenizer=simple_tokenizer
# #     )

# #     # =========================
# #     # 🔥 CLASS WEIGHTS (CRITICAL FIX)
# #     # =========================
# #     genre_labels = []
# #     for i in range(len(dataset)):
# #         item = dataset[i]
# #         if item and "genre" in item:
# #             genre_labels.append(item["genre"].item())

# #     genre_labels = np.array(genre_labels)

# #     class_weights = compute_class_weight(
# #         class_weight="balanced",
# #         classes=np.unique(genre_labels),
# #         y=genre_labels
# #     )

# #     class_weights = torch.tensor(class_weights, dtype=torch.float32)
# #     print(" Genre class weights:", class_weights.tolist())

# #     # =========================
# #     # Split
# #     # =========================
# #     train_size = int(0.9 * len(dataset))
# #     test_size = len(dataset) - train_size
# #     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# #     print(f" Dataset split: {train_size} train | {test_size} test")

# #     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
# #     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# #     # =========================
# #     # Model
# #     # =========================
# #     cfg = ModelConfig(
# #         n_mels=80,
# #         vocab_size=256,
# #         pad_idx=0,
# #         n_genre=len(genre_map),
# #         hidden_size=64,
# #         num_layers=2
# #     )

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = MultiTaskMultimodalLSTM(cfg).to(device)
# #     class_weights = class_weights.to(device)

# #     optimizer = optim.Adam(model.parameters(), lr=1e-4)
# #     scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# #     SAVE_DIR = "Save_model"
# #     os.makedirs(SAVE_DIR, exist_ok=True)

# #     best_loss = float("inf")
# #     EPOCHS = 5

# #     # =========================
# #     # Training
# #     # =========================
# #     for epoch in range(EPOCHS):
# #         model.train()
# #         total_loss = 0.0

# #         for batch in train_loader:
# #             if batch is None:
# #                 continue

# #             mel = batch["mel"].to(device)
# #             mel_lens = batch["mel_lens"].to(device)
# #             tokens = batch["tokens"].to(device)
# #             tok_lens = batch["tok_lens"].to(device)
# #             genre = batch["genre"].to(device)

# #             optimizer.zero_grad()
# #             outputs = model(mel, mel_lens, tokens, tok_lens)

# #             # 🔥 Weighted genre loss
# #             genre_loss = F.cross_entropy(
# #                 outputs["genre"],
# #                 genre,
# #                 weight=class_weights
# #             )

# #             valence_loss = F.mse_loss(outputs["valence"], batch["valence"].to(device))
# #             arousal_loss = F.mse_loss(outputs["arousal"], batch["arousal"].to(device))

# #             loss = valence_loss + arousal_loss + genre_loss
# #             loss.backward()
# #             optimizer.step()

# #             total_loss += loss.item()

# #         scheduler.step()
# #         avg_loss = total_loss / len(train_loader)

# #         print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# #         if avg_loss < best_loss:
# #             best_loss = avg_loss
# #             torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
# #             print("✅ Saved best model")

# #     print("\n Training complete!")





# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# import pandas as pd
# import numpy as np
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import StepLR
# from sklearn.utils.class_weight import compute_class_weight

# from dataset import MusicDataset
# from model import ModelConfig, MultiTaskMultimodalLSTM
# from genre_mapper import MAIN_GENRES


# # =========================
# # Collate function
# # =========================
# def collate_fn(batch):
#     batch = [x for x in batch if x is not None and "genre" in x]
#     if len(batch) == 0:
#         return None

#     mel_lens = [item["mel_len"] for item in batch]
#     mel_padded = pad_sequence([item["mel"] for item in batch], batch_first=True)

#     tok_lens = [item["tok_len"] for item in batch]
#     tokens_padded = pad_sequence(
#         [item["tokens"] for item in batch],
#         batch_first=True,
#         padding_value=0
#     )

#     return {
#         "mel": mel_padded,
#         "mel_lens": torch.tensor(mel_lens),
#         "tokens": tokens_padded,
#         "tok_lens": torch.tensor(tok_lens),
#         "valence": torch.stack([item["valence"] for item in batch]),
#         "arousal": torch.stack([item["arousal"] for item in batch]),
#         "genre": torch.stack([item["genre"] for item in batch]),
#     }


# # =========================
# # Paths (TRAIN ONLY)
# # =========================
# CSV_FILE = "../data/labels_train.csv"
# AUDIO_DIR = "../data/audio"
# LYRICS_DIR = "../data/lyrics"


# # =========================
# # Tokenizer
# # =========================
# def simple_tokenizer(text):
#     return [ord(c) % 256 for c in text]


# # =========================
# # Genre map (FIXED 8 genres)
# # =========================
# genre_map = {g: i for i, g in enumerate(MAIN_GENRES)}
# print("🎵 Training on MAIN genres:")
# print(genre_map)


# # =========================
# # Main Training
# # =========================
# if __name__ == "__main__":

#     dataset = MusicDataset(
#         CSV_FILE,
#         AUDIO_DIR,
#         LYRICS_DIR,
#         genre_map=genre_map,
#         tokenizer=simple_tokenizer
#     )

#     print(f"✅ Training samples: {len(dataset)}")

#     # =========================
#     # 🔥 CLASS WEIGHTS
#     # =========================
#     genre_labels = []
#     for i in range(len(dataset)):
#         item = dataset[i]
#         if item and "genre" in item:
#             genre_labels.append(item["genre"].item())

#     genre_labels = np.array(genre_labels)

#     class_weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.unique(genre_labels),
#         y=genre_labels
#     )

#     class_weights = torch.tensor(class_weights, dtype=torch.float32)
#     print("⚖ Genre class weights:", class_weights.tolist())

#     # =========================
#     # DataLoader
#     # =========================
#     train_loader = DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=True,
#         collate_fn=collate_fn
#     )

#     # =========================
#     # Model
#     # =========================
#     cfg = ModelConfig(
#         n_mels=80,
#         vocab_size=256,
#         pad_idx=0,
#         n_genre=len(genre_map),
#         hidden_size=64,
#         num_layers=2
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultiTaskMultimodalLSTM(cfg).to(device)
#     class_weights = class_weights.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

#     SAVE_DIR = "Save_model"
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     best_loss = float("inf")
#     EPOCHS = 5

#     # =========================
#     # Training loop
#     # =========================
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss = 0.0

#         for batch in train_loader:
#             if batch is None:
#                 continue

#             mel = batch["mel"].to(device)
#             mel_lens = batch["mel_lens"].to(device)
#             tokens = batch["tokens"].to(device)
#             tok_lens = batch["tok_lens"].to(device)
#             genre = batch["genre"].to(device)

#             optimizer.zero_grad()
#             outputs = model(mel, mel_lens, tokens, tok_lens)

#             genre_loss = F.cross_entropy(
#                 outputs["genre"],
#                 genre,
#                 weight=class_weights
#             )

#             valence_loss = F.mse_loss(outputs["valence"], batch["valence"].to(device))
#             arousal_loss = F.mse_loss(outputs["arousal"], batch["arousal"].to(device))

#             loss = valence_loss + arousal_loss + genre_loss
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         scheduler.step()
#         avg_loss = total_loss / len(train_loader)

#         print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
#             print(" Saved best model")

#     print("\n Training complete!")


#train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight

from music_project.dataset import MusicDataset
from music_project.model import ModelConfig, MultiTaskMultimodalLSTM
from music_project.genre_mapper import MAIN_GENRES


# =========================
# PATHS (FROM /content COPY)
# =========================
DRIVE_BASE = "/content/drive/MyDrive/music_project"
BASE_PATH = "/content/music_project"

CSV_FILE = os.path.join(BASE_PATH, "labels_train.csv")
AUDIO_DIR = os.path.join(BASE_PATH, "audio")
LYRICS_DIR = os.path.join(BASE_PATH, "lyrics")

SAVE_DIR = os.path.join(DRIVE_BASE, "Save_model")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")


# =========================
# Collate Function
# =========================
def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    return {
        "mel": torch.stack([item["mel"] for item in batch]),
        "mel_lens": torch.tensor([item["mel_len"] for item in batch]),
        "tokens": pad_sequence(
            [item["tokens"] for item in batch],
            batch_first=True,
            padding_value=0
        ),
        "tok_lens": torch.tensor([item["tok_len"] for item in batch]),
        "genre": torch.stack([item["genre"] for item in batch]),
        "valence": torch.stack([item["valence"] for item in batch]),
        "arousal": torch.stack([item["arousal"] for item in batch]),
    }


# =========================
# Tokenizer
# =========================
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]


# =========================
# Dataset
# =========================
genre_map = {g: i for i, g in enumerate(MAIN_GENRES)}

dataset = MusicDataset(
    csv_file=CSV_FILE,
    audio_dir=AUDIO_DIR,
    lyrics_dir=LYRICS_DIR,
    genre_map=genre_map,
    tokenizer=simple_tokenizer,
    max_len=2600
)

print(f"✅ Training samples: {len(dataset)}")


# =========================
# Class Weights (Smoothed)
# =========================
genre_labels = []
for i in range(len(dataset)):
    item = dataset[i]
    if item is not None:
        genre_labels.append(item["genre"].item())

genre_labels = np.array(genre_labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(genre_labels),
    y=genre_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float32)
class_weights = torch.sqrt(class_weights)  # 🔥 smoothing


# =========================
# DataLoader (OPTIMIZED)
# =========================
train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)


# =========================
# Model Config
# =========================
cfg = ModelConfig(
    n_mels=123,
    n_genre=8,
    vocab_size=256,
    pad_idx=0,
    hidden_size=128,
    num_layers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskMultimodalLSTM(cfg).to(device)
class_weights = class_weights.to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

best_loss = float("inf")
EPOCHS = 30


# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        if batch is None:
            continue


        mel = batch["mel"].to(device)
        mel_lens = batch["mel_lens"].to(device)
        tokens = batch["tokens"].to(device)
        tok_lens = batch["tok_lens"].to(device)
        genre = batch["genre"].to(device)
        valence = batch["valence"].to(device)
        arousal = batch["arousal"].to(device)

        optimizer.zero_grad()

        outputs = model(mel, mel_lens, tokens, tok_lens)

        genre_loss = F.cross_entropy(
            outputs["genre"],
            genre,
            weight=class_weights
        )

        valence_loss = F.mse_loss(outputs["valence"], valence)
        arousal_loss = F.mse_loss(outputs["arousal"], arousal)

        loss = genre_loss + valence_loss + arousal_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save best model to DRIVE
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Saved best model → {MODEL_PATH}")

print("\n🎉 Training complete!")