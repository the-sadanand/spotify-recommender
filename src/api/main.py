"""
FastAPI serving layer for the Spotify Two-Tower Recommender.

Endpoints:
    POST /recommend       — get top-K track recommendations for a playlist
    GET  /metrics         — Prometheus metrics
    GET  /health          — liveness probe
"""

import logging
import os
import pickle
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import mlflow.pytorch
import numpy as np
import faiss
import redis.asyncio as redis
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from scipy.special import softmax
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.two_tower import TwoTowerModel

# ─── Configuration ────────────────────────────────────────────────────────────
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI",      "http://mlflow:5000")
REDIS_HOST        = os.getenv("REDIS_HOST",               "redis")
REDIS_PORT        = int(os.getenv("REDIS_PORT",           "6379"))
CACHE_TTL         = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600"))
ARTIFACTS_DIR     = os.getenv("ARTIFACTS_DIR",            "/app/artifacts")
DRIFT_THRESHOLD   = float(os.getenv("DRIFT_KL_THRESHOLD","0.5"))
DRIFT_WINDOW      = int(os.getenv("DRIFT_WINDOW",         "100"))
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

# ─── Prometheus metrics ───────────────────────────────────────────────────────
LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Latency of POST /recommend requests",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
CACHE_HITS   = Counter("user_embedding_cache_hits_total",   "Redis cache hits")
CACHE_MISSES = Counter("user_embedding_cache_misses_total", "Redis cache misses")
DRIFT_GAUGE  = Gauge("concept_drift_kl_divergence", "Latest KL divergence vs. training dist.")


# ─── Global state ─────────────────────────────────────────────────────────────
class AppState:
    model:              Optional[TwoTowerModel] = None
    track_embeddings:   Optional[np.ndarray]    = None
    training_dist:      Optional[GaussianMixture] = None
    track2idx:          Optional[dict]          = None
    playlist_track_map: Optional[dict]          = None
    redis_client:       Optional[redis.Redis]   = None
    recent_embeddings:  deque                   = deque(maxlen=DRIFT_WINDOW)
    num_tracks:         int                     = 0
    track_index:        Optional[faiss.Index]  = None


state = AppState()


# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading artifacts...")

    # 1. Track embeddings
    emb_path = os.path.join(ARTIFACTS_DIR, "track_embeddings.npy")
    state.track_embeddings = np.load(emb_path).astype(np.float32, copy=False)
    state.num_tracks = len(state.track_embeddings)

    # Track embeddings are L2-normalized by the model, so HNSW L2 search
    # preserves cosine-similarity ranking while providing ANN retrieval.
    state.track_index = faiss.IndexHNSWFlat(
        state.track_embeddings.shape[1], 32
    )
    state.track_index.hnsw.efSearch = 64
    state.track_index.add(state.track_embeddings)
    logger.info(
        f"Loaded track embeddings: shape={state.track_embeddings.shape}, "
        f"FAISS index={state.track_index.ntotal}"
    )

    # 2. Baseline distribution
    dist_path = os.path.join(ARTIFACTS_DIR, "training_track_distribution.pkl")
    with open(dist_path, "rb") as f:
        state.training_dist = pickle.load(f)
    logger.info("Loaded training distribution (GMM)")

    # 3. Vocabularies
    t2i_path = os.path.join(ARTIFACTS_DIR, "track2idx.npy")
    ptm_path = os.path.join(ARTIFACTS_DIR, "playlist_track_map.npy")
    state.track2idx          = np.load(t2i_path, allow_pickle=True).item()
    state.playlist_track_map = np.load(ptm_path, allow_pickle=True).item()
    logger.info(f"Loaded {len(state.track2idx)} tracks, {len(state.playlist_track_map)} playlists")

    # 4. PyTorch model from MLflow
    run_id_path = os.path.join(ARTIFACTS_DIR, "mlflow_run_id.txt")
    try:
        with open(run_id_path) as f:
            run_id = f.read().strip()
        mlflow.set_tracking_uri(MLFLOW_URI)
        model_uri = f"runs:/{run_id}/two-tower-recommender"
        state.model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        state.model.eval()
        logger.info(f"Loaded model from MLflow run {run_id}")
    except Exception as e:
        logger.exception("Failed to load model from MLflow")
        raise RuntimeError(
            f"Could not load model from MLflow: {e}"
        ) from e

    # 5. Redis
    state.redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False
    )
    try:
        await state.redis_client.ping()
        logger.info("Redis connection OK")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    yield
    if state.redis_client is not None:
        await state.redis_client.aclose()
    logger.info("Shutting down.")


app = FastAPI(
    title="Spotify Two-Tower Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class RecommendRequest(BaseModel):
    playlist_id: int
    top_k: int = 10


class RecommendResponse(BaseModel):
    recommended_track_ids: list[int]
    retrieved_from_cache: bool


# ─── Helpers ──────────────────────────────────────────────────────────────────
CACHE_KEY_PREFIX = "playlist_embedding:"


async def _get_playlist_embedding(playlist_id: int) -> tuple[np.ndarray, bool]:
    """Retrieve playlist embedding — from cache or computed fresh."""
    cache_key = f"{CACHE_KEY_PREFIX}{playlist_id}"
    redis_val = await state.redis_client.get(cache_key)

    if redis_val is not None:
        CACHE_HITS.inc()
        emb = np.frombuffer(redis_val, dtype=np.float32)
        return emb, True

    CACHE_MISSES.inc()

    # Compute embedding via playlist tower
    track_ids_raw = state.playlist_track_map.get(playlist_id)
    if not track_ids_raw:
        raise HTTPException(status_code=404, detail="Playlist not found")

    # Shift IDs by +1 (same as training), cap at max_context=20
    context = [t + 1 for t in track_ids_raw[:20]]
    length  = len(context)
    padded  = context + [0] * (20 - length)

    pt = torch.tensor([padded], dtype=torch.long).to(DEVICE)
    pl = torch.tensor([length], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        emb_tensor = state.model.encode_playlist(pt, pl)  # (1, output_dim)
    emb = emb_tensor.cpu().numpy().squeeze().astype(np.float32)

    # Store in Redis with TTL
    await state.redis_client.setex(cache_key, CACHE_TTL, emb.tobytes())
    return emb, False


def _find_top_k(playlist_emb: np.ndarray, top_k: int) -> list[int]:
    """Retrieve top-K tracks with the FAISS HNSW ANN index."""
    if top_k <= 0 or state.track_index is None:
        return []

    k = min(top_k, state.num_tracks)
    query = np.asarray(playlist_emb, dtype=np.float32).reshape(1, -1)
    _, indices = state.track_index.search(query, k)
    return indices[0].tolist()


def _check_drift(embedding: np.ndarray):
    """Add embedding to buffer; check KL divergence when buffer is full."""
    state.recent_embeddings.append(embedding)
    if len(state.recent_embeddings) < DRIFT_WINDOW:
        return

    recent = np.stack(list(state.recent_embeddings))

    # Fit a GMM on recent embeddings
    n_comp = min(state.training_dist.n_components, len(recent))
    try:
        recent_gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="diag",
            max_iter=100,
            random_state=0,
        )
        recent_gmm.fit(recent)
    except Exception:
        return

    # Monte-Carlo KL divergence: KL(recent || training)
    # Sample from recent, evaluate log-prob under both
    n_samples = 1000
    samples = recent_gmm.sample(n_samples)[0]
    log_p = recent_gmm.score_samples(samples)      # log P(x)
    log_q = state.training_dist.score_samples(samples)  # log Q(x)
    kl_div = float(np.mean(log_p - log_q))

    DRIFT_GAUGE.set(kl_div)
    logger.info(f"KL divergence (recent vs training): {kl_div:.4f}")

    if kl_div > DRIFT_THRESHOLD:
        logger.warning(
            f"DRIFT DETECTED: KL divergence={kl_div:.4f} exceeds threshold={DRIFT_THRESHOLD}"
        )

    # Clear buffer after check
    state.recent_embeddings.clear()


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    start = time.perf_counter()

    if req.playlist_id not in state.playlist_track_map:
        raise HTTPException(status_code=404, detail="Playlist not found")

    playlist_emb, from_cache = await _get_playlist_embedding(req.playlist_id)
    top_indices = _find_top_k(playlist_emb, req.top_k)

    # Drift detection (async-style: don't block response)
    try:
        _check_drift(playlist_emb)
    except Exception as e:
        logger.debug(f"Drift check error (non-fatal): {e}")

    elapsed = time.perf_counter() - start
    LATENCY.observe(elapsed)

    return RecommendResponse(
        recommended_track_ids=top_indices,
        retrieved_from_cache=from_cache,
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
async def health():
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    if state.redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Redis not initialized"
        )

    try:
        await state.redis_client.ping()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Redis unavailable"
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "redis_connected": True,
    }
