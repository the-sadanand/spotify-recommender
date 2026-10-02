"""
Data preprocessing for the Spotify Million Playlist Dataset.
Builds track/playlist vocabularies and creates training pairs.
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np


def load_playlists(data_dir: str, max_playlists: int = 1000) -> Iterator[dict]:
    """Load playlists from the Spotify MPD JSON files."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        print(f"[WARNING] No JSON files found in {data_dir}. Generating synthetic data for demo.")
        yield from _generate_synthetic_data(max_playlists)
        return

    loaded = 0
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        for playlist in data.get("playlists", []):
            yield playlist
            loaded += 1
            if loaded >= max_playlists:
                return

    print(f"Streamed {loaded} playlists from {data_dir}")


def _generate_synthetic_data(num_playlists: int = 1000) -> List[dict]:
    """Generate synthetic playlist data for demo/testing."""
    num_tracks = 5000
    rng = random.Random(42)
    for pid in range(num_playlists):
        playlist_len = rng.randint(5, 30)
        tracks = []
        track_ids = rng.sample(range(num_tracks), min(playlist_len, num_tracks))
        for tid in track_ids:
            tracks.append({
                "track_uri": f"spotify:track:synth_{tid:06d}",
                "track_name": f"Track {tid}",
                "artist_name": f"Artist {tid % 500}",
            })
        playlists.append({
            "pid": pid,
            "name": f"Playlist {pid}",
            "tracks": tracks,
        })
    print(f"Generated {num_playlists} synthetic playlists with up to 5000 tracks.")
    return playlists


def build_vocabularies(playlists: List[dict]) -> Tuple[Dict, Dict]:
    """
    Build track-to-index and playlist-to-index vocabularies.
    Returns (track2idx, playlist2idx)
    """
    track2idx: Dict[str, int] = {}
    playlist2idx: Dict[int, int] = {}

    for playlist in playlists:
        pid = playlist["pid"]
        if pid not in playlist2idx:
            playlist2idx[pid] = len(playlist2idx)

        for track in playlist.get("tracks", []):
            uri = track["track_uri"]
            if uri not in track2idx:
                track2idx[uri] = len(track2idx)

    print(f"Vocabulary: {len(track2idx)} unique tracks, {len(playlist2idx)} playlists")
    return track2idx, playlist2idx


def create_playlist_track_map(
    playlists: List[dict], track2idx: Dict[str, int]
) -> Dict[int, List[int]]:
    """Map playlist_id -> list of track indices."""
    playlist_track_map: Dict[int, List[int]] = defaultdict(list)
    for playlist in playlists:
        pid = playlist["pid"]
        for track in playlist.get("tracks", []):
            uri = track["track_uri"]
            if uri in track2idx:
                playlist_track_map[pid].append(track2idx[uri])
    return dict(playlist_track_map)


def generate_training_pairs(
    playlist_track_map: Dict[int, List[int]],
    num_tracks: int,
    num_negatives: int = 4,
    split: str = "train",
    val_mod: int = 10,
) -> Iterator[Tuple[List[int], int, int]]:
    """Stream training triples without materializing the complete dataset."""
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    all_track_ids = list(range(num_tracks))

    for pid, track_ids in playlist_track_map.items():
        is_val = pid % val_mod == 0
        if (split == "val") != is_val or len(track_ids) < 2:
            continue

        playlist_set = set(track_ids)
        for pos_track in track_ids:
            context = [t for t in track_ids if t != pos_track]
            if not context:
                continue

            negatives = []
            attempts = 0
            while len(negatives) < num_negatives and attempts < 100:
                neg = random.choice(all_track_ids)
                if neg not in playlist_set:
                    negatives.append(neg)
                attempts += 1

            for neg in negatives:
                yield (context[:20], pos_track, neg)


def save_vocab(track2idx: Dict, playlist_track_map: Dict, artifacts_dir: str):
    """Save vocabularies to disk for serving time."""
    os.makedirs(artifacts_dir, exist_ok=True)
    np.save(os.path.join(artifacts_dir, "track2idx.npy"), track2idx)
    np.save(os.path.join(artifacts_dir, "playlist_track_map.npy"), playlist_track_map)
    print(f"Saved vocabularies to {artifacts_dir}")
