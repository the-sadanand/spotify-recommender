"""
Two-Tower Neural Network for Spotify track recommendation.

Architecture:
  - Playlist Tower: Embeds a sequence of track IDs, averages them, passes through MLP
  - Track Tower: Embeds a single track ID, passes through MLP
  - Training: Contrastive / BPR loss pushing positives close, negatives far
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaylistTower(nn.Module):
    """
    Encodes a playlist context (bag of track embeddings) into a dense vector.
    Input: (batch, seq_len) track IDs with padding=0
    Output: (batch, output_dim) L2-normalized embedding
    """

    def __init__(self, num_tracks: int, embed_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        # +1 for padding index 0
        self.embedding = nn.Embedding(num_tracks + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, output_dim),
        )

    def forward(self, track_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        track_ids: (batch, max_seq_len) — padded with 0
        lengths:   (batch,) — actual sequence lengths
        """
        embedded = self.embedding(track_ids)  # (B, L, embed_dim)
        # Mean pooling over non-padded positions
        mask = (track_ids != 0).float().unsqueeze(-1)  # (B, L, 1)
        summed = (embedded * mask).sum(dim=1)           # (B, embed_dim)
        lengths_clamped = lengths.float().clamp(min=1).unsqueeze(-1)
        pooled = summed / lengths_clamped               # (B, embed_dim)
        out = self.mlp(pooled)
        return F.normalize(out, dim=-1)


class TrackTower(nn.Module):
    """
    Encodes a single track ID into a dense vector.
    Input: (batch,) track IDs
    Output: (batch, output_dim) L2-normalized embedding
    """

    def __init__(self, num_tracks: int, embed_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_tracks + 1, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, output_dim),
        )

    def forward(self, track_id: torch.Tensor) -> torch.Tensor:
        """track_id: (batch,)"""
        embedded = self.embedding(track_id)  # (B, embed_dim)
        out = self.mlp(embedded)
        return F.normalize(out, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full two-tower model combining playlist tower and track tower.
    """

    def __init__(
        self,
        num_tracks: int,
        embed_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.playlist_tower = PlaylistTower(num_tracks, embed_dim, output_dim, dropout)
        self.track_tower = TrackTower(num_tracks, embed_dim, output_dim, dropout)

    def forward(
        self,
        playlist_tracks: torch.Tensor,
        playlist_lengths: torch.Tensor,
        track_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (playlist_embedding, track_embedding) — both L2 normalized.
        """
        playlist_emb = self.playlist_tower(playlist_tracks, playlist_lengths)
        track_emb = self.track_tower(track_ids)
        return playlist_emb, track_emb

    def encode_playlist(
        self, playlist_tracks: torch.Tensor, playlist_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode only the playlist side (for inference)."""
        return self.playlist_tower(playlist_tracks, playlist_lengths)

    def encode_track(self, track_ids: torch.Tensor) -> torch.Tensor:
        """Encode only a track (for pre-computing all track embeddings)."""
        return self.track_tower(track_ids)


def bpr_loss(
    playlist_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.
    Maximises score(playlist, pos) - score(playlist, neg).
    """
    pos_score = (playlist_emb * pos_emb).sum(dim=-1)   # (B,)
    neg_score = (playlist_emb * neg_emb).sum(dim=-1)   # (B,)
    loss = -F.logsigmoid(pos_score - neg_score).mean()
    return loss


def accuracy(
    playlist_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> float:
    """Fraction of pairs where positive is scored higher than negative."""
    pos_score = (playlist_emb * pos_emb).sum(dim=-1)
    neg_score = (playlist_emb * neg_emb).sum(dim=-1)
    return (pos_score > neg_score).float().mean().item()
