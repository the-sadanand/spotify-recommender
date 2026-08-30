"""
PyTorch Dataset for Two-Tower training triples.
"""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class PlaylistTrackDataset(Dataset):
    """
    Dataset of (context_track_ids, positive_track_id, negative_track_id) triples.
    Handles variable-length context via padding.
    """

    def __init__(
        self,
        triples: List[Tuple[List[int], int, int]],
        max_context_len: int = 20,
    ):
        self.triples = triples
        self.max_context_len = max_context_len

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        context, pos_track, neg_track = self.triples[idx]

        # Shift all IDs by +1 to reserve 0 for padding
        context_shifted = [t + 1 for t in context[: self.max_context_len]]
        length = len(context_shifted)

        # Pad context to max_context_len
        padded = context_shifted + [0] * (self.max_context_len - length)

        return {
            "playlist_tracks": torch.tensor(padded, dtype=torch.long),
            "playlist_length": torch.tensor(length, dtype=torch.long),
            "pos_track": torch.tensor(pos_track + 1, dtype=torch.long),
            "neg_track": torch.tensor(neg_track + 1, dtype=torch.long),
        }
