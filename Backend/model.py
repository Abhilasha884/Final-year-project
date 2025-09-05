import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


# --------------------------
# Config
# --------------------------
class ModelConfig:
    def __init__(self, n_mels, n_emotion, vocab_size, pad_idx,
                 n_genre=None, hidden_size=128, num_layers=2):
        self.n_mels = n_mels
        self.n_emotion = n_emotion
        self.n_genre = n_genre
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.num_layers = num_layers


# --------------------------
# Audio Encoder (with pack_padded_sequence)
# --------------------------
class AudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.n_mels,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, lengths):
        # x: [B, T, n_mels], lengths: [B]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # Concatenate forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=-1)  # [B, 2*hidden_size]
        return h


# --------------------------
# Lyrics Encoder (with pack_padded_sequence)
# --------------------------
class LyricsEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, embed_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, tokens, lengths):
        emb = self.embedding(tokens)  # [B, T, E]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # [B, 2*hidden_size]
        return h


# --------------------------
# Multimodal Fusion + Multi-task heads
# --------------------------
class MultiTaskMultimodalLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.audio_encoder = AudioEncoder(cfg)
        self.lyrics_encoder = LyricsEncoder(cfg)

        fusion_dim = 2 * cfg.hidden_size * 2  # audio (2*hidden) + lyrics (2*hidden)

        # Task heads
        self.fc_emotion = nn.Linear(fusion_dim, cfg.n_emotion)
        self.fc_valence = nn.Linear(fusion_dim, 1)
        self.fc_arousal = nn.Linear(fusion_dim, 1)
        self.fc_genre = nn.Linear(fusion_dim, cfg.n_genre) if cfg.n_genre else None

    def forward(self, mel, mel_lens, tokens, tok_lens):
        audio_repr = self.audio_encoder(mel, mel_lens)      # [B, 2H]
        lyrics_repr = self.lyrics_encoder(tokens, tok_lens) # [B, 2H]

        fused = torch.cat([audio_repr, lyrics_repr], dim=-1)  # [B, 4H]

        return {
            "emotion": self.fc_emotion(fused),
            "valence": self.fc_valence(fused).squeeze(-1),
            "arousal": self.fc_arousal(fused).squeeze(-1),
            "genre": self.fc_genre(fused) if self.fc_genre else None,
        }


# --------------------------
# Loss function
# --------------------------
def compute_multitask_loss(outputs, targets):
    losses = {}
    total_loss = 0.0

    # Emotion classification
    losses["emotion"] = F.cross_entropy(outputs["emotion"], targets["emotion"])
    total_loss += losses["emotion"]

    # Valence regression
    losses["valence"] = F.mse_loss(outputs["valence"], targets["valence"])
    total_loss += losses["valence"]

    # Arousal regression
    losses["arousal"] = F.mse_loss(outputs["arousal"], targets["arousal"])
    total_loss += losses["arousal"]

    # Genre classification (if available)
    if outputs["genre"] is not None and "genre" in targets:
        losses["genre"] = F.cross_entropy(outputs["genre"], targets["genre"])
        total_loss += losses["genre"]

    return total_loss, losses
