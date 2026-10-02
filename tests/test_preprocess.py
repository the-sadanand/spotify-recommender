import pytest

from src.training.preprocess import generate_training_pairs, load_playlists


def test_load_playlists_is_streaming():
    playlists = load_playlists("/tmp/path-that-does-not-exist", 5)
    assert not isinstance(playlists, list)
    assert len(list(playlists)) == 5


def test_training_pairs_are_streamed():
    playlist_map = {0: [0, 1, 2], 1: [2, 3, 4]}
    pairs = generate_training_pairs(playlist_map, 10, num_negatives=2)

    assert not isinstance(pairs, list)
    assert len(list(pairs)) > 0


def test_train_and_validation_playlists_do_not_overlap():
    playlist_map = {i: [0, 1, 2] for i in range(20)}
    train = list(generate_training_pairs(playlist_map, 10, 1, split="train"))
    val = list(generate_training_pairs(playlist_map, 10, 1, split="val"))

    train_pos = {pos for _, pos, _ in train}
    val_pos = {pos for _, pos, _ in val}

    assert train
    assert val
    assert train_pos.isdisjoint(val_pos)


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
        list(generate_training_pairs({0: [0, 1]}, 5, split="test"))
