import torch

from src.training.dataset import PlaylistTrackDataset


def test_dataset_encodes_and_pads_triples():
    dataset = PlaylistTrackDataset(
        lambda: iter([([2, 4], 6, 8)]),
        max_context_len=4,
    )

    batch = next(iter(dataset))

    assert torch.equal(
        batch["playlist_tracks"],
        torch.tensor([3, 5, 0, 0]),
    )
    assert batch["playlist_length"].item() == 2
    assert batch["pos_track"].item() == 7
    assert batch["neg_track"].item() == 9
