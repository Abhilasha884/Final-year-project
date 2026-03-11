# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence
# import torch.nn.functional as F

# class ModelConfig:
#     def __init__(self, n_mels, vocab_size, pad_idx,
#                  n_genre=None, hidden_size=64, num_layers=2):
#         self.n_mels = n_mels
#         self.n_genre = n_genre
#         self.vocab_size = vocab_size
#         self.pad_idx = pad_idx
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

# class AudioEncoder(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=cfg.n_mels,
#             hidden_size=cfg.hidden_size,
#             num_layers=cfg.num_layers,
#             batch_first=True,
#             bidirectional=True,
#         )

#     def forward(self, x, lengths):
#         packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (h, _) = self.lstm(packed)
#         # h shape: (num_layers * num_directions, batch, hidden_size)
#         # take last layer forward and backward
#         h_fwd = h[-2]
#         h_bwd = h[-1]
#         h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2*hidden_size]
#         return h

# class LyricsEncoder(nn.Module):
#     def __init__(self, cfg: ModelConfig, embed_dim=128):
#         super().__init__()
#         self.embedding = nn.Embedding(cfg.vocab_size, embed_dim, padding_idx=cfg.pad_idx)
#         self.lstm = nn.LSTM(
#             input_size=embed_dim,
#             hidden_size=cfg.hidden_size,
#             num_layers=cfg.num_layers,
#             batch_first=True,
#             bidirectional=True,
#         )

#     def forward(self, tokens, lengths):
#         emb = self.embedding(tokens)
#         packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (h, _) = self.lstm(packed)
#         h_fwd = h[-2]
#         h_bwd = h[-1]
#         h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2*hidden_size]
#         return h

# class MultiTaskMultimodalLSTM(nn.Module):
#     def __init__(self, cfg: ModelConfig):
#         super().__init__()
#         self.audio_encoder = AudioEncoder(cfg)
#         self.lyrics_encoder = LyricsEncoder(cfg)

#         fusion_dim = 2 * cfg.hidden_size * 2  # audio (2*hidden) + lyrics (2*hidden)

#         # Heads
#         self.fc_valence = nn.Linear(fusion_dim, 1)
#         self.fc_arousal = nn.Linear(fusion_dim, 1)
#         self.fc_genre = nn.Linear(fusion_dim, cfg.n_genre) if cfg.n_genre else None

#     def forward(self, mel, mel_lens, tokens, tok_lens):
#         audio_repr = self.audio_encoder(mel, mel_lens)        # [B, 2*H]
#         lyrics_repr = self.lyrics_encoder(tokens, tok_lens)   # [B, 2*H]
#         fused = torch.cat([audio_repr, lyrics_repr], dim=-1)  # [B, 4*H]

#         valence = self.fc_valence(fused).squeeze(-1)
#         arousal = self.fc_arousal(fused).squeeze(-1)
#         genre = self.fc_genre(fused) if self.fc_genre is not None else None

#         return {
#             "valence": valence,
#             "arousal": arousal,
#             "genre": genre,
#             "embedding": fused   # expose fused embedding for indexing
#         }

# def compute_multitask_loss(outputs, targets, weights=None):
#     losses = {}
#     total_loss = 0.0

#     # Valence regression
#     losses["valence"] = F.mse_loss(outputs["valence"], targets["valence"])
#     total_loss += losses["valence"]

#     # Arousal regression
#     losses["arousal"] = F.mse_loss(outputs["arousal"], targets["arousal"])
#     total_loss += losses["arousal"]

#     # Genre classification 
#     if outputs["genre"] is not None and "genre" in targets:
#         losses["genre"] = F.cross_entropy(outputs["genre"], targets["genre"])
#         total_loss += losses["genre"]

#     return total_loss, losses




#model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


# =====================================================
# Model Config
# =====================================================
class ModelConfig:
    def __init__(self, n_mels, vocab_size, pad_idx,
                 n_genre=None, hidden_size=128, num_layers=2):
        self.n_mels = n_mels
        self.n_genre = n_genre
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.num_layers = num_layers


# =====================================================
# Audio Encoder (CNN + BiLSTM)
# =====================================================
class AudioEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers

        # CNN block
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # We will initialize LSTM later dynamically
        self.lstm = nn.LSTM(
            input_size=1920,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True
        )
        # self.lstm = None

    def forward(self, x, lengths):

        B, T, F = x.shape

        x = x.transpose(1, 2)   # [B, F, T]
        x = x.unsqueeze(1)      # [B, 1, F, T]

        x = self.cnn(x)

        B, C, F_new, T_new = x.shape

        x = x.permute(0, 3, 1, 2)   # [B, T_new, C, F_new]
        x = x.reshape(B, T_new, C * F_new)

        lengths = lengths // 4
        lengths = torch.clamp(lengths, min=1)

        # # 🔥 Dynamically create LSTM based on real size
        # if self.lstm is None:
        #     self.lstm = nn.LSTM(
        #         input_size=C * F_new,
        #         hidden_size=self.hidden_size,
        #         num_layers=self.num_layers,
        #         batch_first=True,
        #         bidirectional=True,
        #     ).to(x.device)

        self.lstm.flatten_parameters()

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)

        h_fwd = h[-2]
        h_bwd = h[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)

        return h

# =====================================================
# Lyrics Encoder (BiLSTM)
# =====================================================
class LyricsEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            cfg.vocab_size,
            embed_dim,
            padding_idx=cfg.pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, tokens, lengths):

        emb = self.embedding(tokens)

        lengths = torch.clamp(lengths, min=1)

        self.lstm.flatten_parameters()

        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)

        h_fwd = h[-2]
        h_bwd = h[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2H]

        return h


# =====================================================
# Multimodal Model
# =====================================================
class MultiTaskMultimodalLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.audio_encoder = AudioEncoder(cfg)
        self.lyrics_encoder = LyricsEncoder(cfg)

        # audio (2H) + lyrics (2H)
        fusion_dim = 4 * cfg.hidden_size

        self.fc_valence = nn.Linear(fusion_dim, 1)
        self.fc_arousal = nn.Linear(fusion_dim, 1)
        self.fc_genre = nn.Linear(fusion_dim, cfg.n_genre) if cfg.n_genre else None

    def forward(self, mel, mel_lens, tokens, tok_lens):

        audio_repr = self.audio_encoder(mel, mel_lens)        # [B, 2H]
        lyrics_repr = self.lyrics_encoder(tokens, tok_lens)   # [B, 2H]

        fused = torch.cat([audio_repr, lyrics_repr], dim=-1)  # [B, 4H]

        valence = self.fc_valence(fused).squeeze(-1)
        arousal = self.fc_arousal(fused).squeeze(-1)
        genre = self.fc_genre(fused) if self.fc_genre is not None else None

        return {
            "valence": valence,
            "arousal": arousal,
            "genre": genre,
            "embedding": fused
        }


# =====================================================
# Optional Multitask Loss
# =====================================================
def compute_multitask_loss(outputs, targets):

    losses = {}
    total_loss = 0.0

    losses["valence"] = F.mse_loss(outputs["valence"], targets["valence"])
    total_loss += losses["valence"]

    losses["arousal"] = F.mse_loss(outputs["arousal"], targets["arousal"])
    total_loss += losses["arousal"]

    if outputs["genre"] is not None and "genre" in targets:
        losses["genre"] = F.cross_entropy(
            outputs["genre"],
            targets["genre"]
        )
        total_loss += losses["genre"]

    return total_loss, losses