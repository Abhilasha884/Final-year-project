# import os
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torch.nn.utils.rnn import pad_sequence
# import pandas as pd
# from dataset import MusicDataset
# from model import ModelConfig, MultiTaskMultimodalLSTM, compute_multitask_loss
# from torch.optim.lr_scheduler import StepLR


# # Collate function (padding)

# def collate_fn(batch):
#     batch = [x for x in batch if x is not None]
#     if len(batch) == 0:
#         return None

#     mel_lens = [item["mel_len"] for item in batch]
#     mel_padded = pad_sequence([item["mel"] for item in batch], batch_first=True)

#     tok_lens = [item["tok_len"] for item in batch]
#     tokens_padded = pad_sequence([item["tokens"] for item in batch],
#                                  batch_first=True, padding_value=0)

#     valences = torch.stack([item["valence"] for item in batch])
#     arousals = torch.stack([item["arousal"] for item in batch])

#     genres = None
#     if "genre" in batch[0]:
#         genres = torch.stack([item["genre"] for item in batch])

#     result = {
#         "mel": mel_padded,
#         "mel_lens": torch.tensor(mel_lens),
#         "tokens": tokens_padded,
#         "tok_lens": torch.tensor(tok_lens),
#         "valence": valences,
#         "arousal": arousals,
#     }
#     if genres is not None:
#         result["genre"] = genres
#     return result


# CSV_FILE = "../data/labels.csv"
# AUDIO_DIR = "../data/audio"
# LYRICS_DIR = "../data/lyrics"


# df = pd.read_csv(CSV_FILE, encoding="utf-8")
# have_genre = "genre" in df.columns and df["genre"].notna().any()
# genre_map = {g: i for i, g in enumerate(sorted(df["genre"].dropna().unique()))} if have_genre else None
# print("Genres:", genre_map if genre_map else None)

# # --------------------------
# # Dummy tokenizer
# # --------------------------
# def simple_tokenizer(text):
#     return [ord(c) % 256 for c in text]


# # Dataset splitting (90% train / 10% test)

# full_dataset = MusicDataset(
#     CSV_FILE, AUDIO_DIR, LYRICS_DIR,
#     genre_map=genre_map,
#     tokenizer=simple_tokenizer
# )

# train_size = int(0.9 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# print(f" Dataset split: {train_size} training samples, {test_size} testing samples")

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


# # Model setup

# vocab_size = 256
# pad_idx = 0
# cfg = ModelConfig(
#     n_mels=80,
#     n_genre=len(genre_map) if have_genre else None,
#     vocab_size=vocab_size,
#     pad_idx=pad_idx,
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MultiTaskMultimodalLSTM(cfg).to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# # --------------------------
# # Save directory
# # --------------------------
# SAVE_DIR = os.path.join(os.path.dirname(__file__), "Save_model")
# os.makedirs(SAVE_DIR, exist_ok=True)
# best_loss = float("inf")

# # --------------------------
# # Training loop
# # --------------------------
# EPOCHS = 5
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0.0

#     for batch in train_loader:
#         if batch is None:
#             continue

#         mel_spec = batch["mel"].to(device)
#         mel_lens = batch["mel_lens"].to(device)
#         tokens = batch["tokens"].to(device)
#         tok_lens = batch["tok_lens"].to(device)

#         optimizer.zero_grad()
#         outputs = model(mel_spec, mel_lens, tokens, tok_lens)

#         targets = {
#             "valence": batch["valence"].to(device),
#             "arousal": batch["arousal"].to(device),
#         }
#         if "genre" in batch:
#             targets["genre"] = batch["genre"].to(device)

#         loss, _ = compute_multitask_loss(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     scheduler.step()
#     avg_loss = total_loss / len(train_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {avg_loss:.4f}")

#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         save_path = os.path.join(SAVE_DIR, "best_model.pth")
#         torch.save(model.state_dict(), save_path)
#         print(f"✅ Saved new best model at {save_path} (loss={best_loss:.4f})")

# print("\n Training complete!")

# # Testing / Evaluation

# print("\n Starting model evaluation on test set...")

# best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
# if os.path.exists(best_model_path):
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     print(f" Loaded best model from {best_model_path}")
# else:
#     print("⚠️ No saved model found, evaluating current weights instead.")

# model.eval()
# test_loss = 0.0
# valence_preds, valence_true = [], []
# arousal_preds, arousal_true = [], []

# with torch.no_grad():
#     for batch in test_loader:
#         if batch is None:
#             continue

#         mel_spec = batch["mel"].to(device)
#         mel_lens = batch["mel_lens"].to(device)
#         tokens = batch["tokens"].to(device)
#         tok_lens = batch["tok_lens"].to(device)

#         outputs = model(mel_spec, mel_lens, tokens, tok_lens)

#         targets = {
#             "valence": batch["valence"].to(device),
#             "arousal": batch["arousal"].to(device),
#         }
#         if "genre" in batch:
#             targets["genre"] = batch["genre"].to(device)

#         loss, _ = compute_multitask_loss(outputs, targets)
#         test_loss += loss.item()

#         valence_preds.extend(outputs["valence"].cpu().tolist())
#         valence_true.extend(targets["valence"].cpu().tolist())
#         arousal_preds.extend(outputs["arousal"].cpu().tolist())
#         arousal_true.extend(targets["arousal"].cpu().tolist())

# avg_test_loss = test_loss / len(test_loader)
# print(f" Test Loss: {avg_test_loss:.4f}")

# valence_mae = sum(abs(p - t) for p, t in zip(valence_preds, valence_true)) / len(valence_true)
# arousal_mae = sum(abs(p - t) for p, t in zip(arousal_preds, arousal_true)) / len(arousal_true)
# print(f" Valence MAE: {valence_mae:.4f} | Arousal MAE: {arousal_mae:.4f}")
# print(" Evaluation complete!")
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.optim.lr_scheduler import StepLR

from dataset import MusicDataset
from model import ModelConfig, MultiTaskMultimodalLSTM, compute_multitask_loss
from genre_mapper import MAIN_GENRES


# =========================
# Collate function
# =========================
def collate_fn(batch):
    # keep only samples that have genre
    batch = [x for x in batch if x is not None and "genre" in x]

    if len(batch) == 0:
        return None

    mel_lens = [item["mel_len"] for item in batch]
    mel_padded = pad_sequence(
        [item["mel"] for item in batch],
        batch_first=True
    )

    tok_lens = [item["tok_len"] for item in batch]
    tokens_padded = pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=0
    )

    result = {
        "mel": mel_padded,
        "mel_lens": torch.tensor(mel_lens),
        "tokens": tokens_padded,
        "tok_lens": torch.tensor(tok_lens),
        "valence": torch.stack([item["valence"] for item in batch]),
        "arousal": torch.stack([item["arousal"] for item in batch]),
        "genre": torch.stack([item["genre"] for item in batch]),
    }

    return result



# =========================
# Paths
# =========================
CSV_FILE = "../data/labels.csv"
AUDIO_DIR = "../data/audio"
LYRICS_DIR = "../data/lyrics"


# =========================
# Simple tokenizer
# =========================
def simple_tokenizer(text):
    return [ord(c) % 256 for c in text]


# =========================
# Load CSV & genre map
# =========================
df = pd.read_csv(CSV_FILE, encoding="utf-8")
have_genre = "genre" in df.columns and df["genre"].notna().any()

# 🔥 MAIN GENRE MAP ONLY
genre_map = {g: i for i, g in enumerate(MAIN_GENRES)} if have_genre else None

print("🎵 Training on MAIN genres only:")
print(genre_map)


# =========================
# Main
# =========================
if __name__ == "__main__":

    # Dataset
    full_dataset = MusicDataset(
        CSV_FILE,
        AUDIO_DIR,
        LYRICS_DIR,
        genre_map=genre_map,
        tokenizer=simple_tokenizer
    )

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size]
    )

    print(f"📊 Dataset split: {train_size} train | {test_size} test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # =========================
    # Model setup
    # =========================
    cfg = ModelConfig(
        n_mels=80,
        vocab_size=256,
        pad_idx=0,
        n_genre=len(genre_map) if have_genre else None,
        hidden_size=64,        # ✅ FINAL
        num_layers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskMultimodalLSTM(cfg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # =========================
    # Save directory
    # =========================
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "Save_model")
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_loss = float("inf")

    # =========================
    # Training loop
    # =========================
    EPOCHS = 5

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

            optimizer.zero_grad()

            outputs = model(mel, mel_lens, tokens, tok_lens)

            targets = {
                "valence": batch["valence"].to(device),
                "arousal": batch["arousal"].to(device),
            }

            if "genre" in batch:
                targets["genre"] = batch["genre"].to(device)

            loss, losses = compute_multitask_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Components: {losses}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model → {save_path}")

    print("\n🎉 Training complete!")


    # =========================
    # Evaluation
    # =========================
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("🔥 Loaded best model for evaluation")

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            mel = batch["mel"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            tokens = batch["tokens"].to(device)
            tok_lens = batch["tok_lens"].to(device)

            outputs = model(mel, mel_lens, tokens, tok_lens)

            targets = {
                "valence": batch["valence"].to(device),
                "arousal": batch["arousal"].to(device),
            }

            if "genre" in batch:
                targets["genre"] = batch["genre"].to(device)

            loss, _ = compute_multitask_loss(outputs, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"📉 Test Loss: {avg_test_loss:.4f}")
