import numpy as np

from src.api.main import state, _find_top_k


def test_find_top_k_uses_faiss_index():
    import faiss

    embeddings = np.eye(3, dtype=np.float32)
    index = faiss.IndexHNSWFlat(3, 16)
    index.add(embeddings)

    state.track_embeddings = embeddings
    state.track_index = index
    state.num_tracks = 3

    assert _find_top_k(embeddings[1], 2)[0] == 1
