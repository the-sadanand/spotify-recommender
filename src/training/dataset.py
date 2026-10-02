"""
PyTorch Dataset for Two-Tower training triples.
"""

from typing import Callable, Iterator, List, Tuple
import torch
from torch.utils.data import IterableDataset


class PlaylistTrackDataset(IterableDataset):
    """Streaming Dataset backed by a triple-generator factory."""

    def __init__(
        self,
        triple_factory: Callable[[], Iterator[Tuple[List[int], int, int]]],
        max_context_len: int = 20,
    ):
        super().__init__()
        self.triple_factory = triple_factory
        self.max_context_len = max_context_len

    def __iter__(self):
        for context, pos_track, neg_track in self.triple_factory():
            context_shifted = [t + 1 for t in context[: self.max_context_len]]
            length = len(context_shifted)
            padded = context_shifted + [0] * (self.max_context_len - length)

            yield {
                "playlist_tracks": torch.tensor(padded, dtype=torch.long),
                "playlist_length": torch.tensor(length, dtype=torch.long),
                "pos_track": torch.tensor(pos_track + 1, dtype=torch.long),
                "neg_track": torch.tensor(neg_track + 1, dtype=torch.long),
            }
