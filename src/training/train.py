"""
End-to-end training script for the Two-Tower Spotify Recommender.

Usage:
    python src/training/train.py

Environment variables:
    MLFLOW_TRACKING_URI  - required MLflow server URL
    DATA_DIR             - path to Spotify MPD JSON files (default: data/)
    ARTIFACTS_DIR        - where to save track_embeddings.npy etc. (default: artifacts/)
    MAX_PLAYLISTS        - number of playlists to use (default: 1000)
    EMBEDDING_DIM        - embedding size (default: 64)
    OUTPUT_DIM           - two-tower output dimension (default: 64)
    LEARNING_RATE        - Adam LR (default: 1e-3)
    BATCH_SIZE           - training batch size (default: 256)
    NUM_EPOCHS           - training epochs (default: 10)
    NUM_NEGATIVES        - negatives per positive (default: 4)
    GMM_COMPONENTS       - components for the baseline GMM (default: 8)
"""

import os
import pickle
import random
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ── resolve src/ on the path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.two_tower import TwoTowerModel, accuracy, bpr_loss
from src.training.dataset import PlaylistTrackDataset
from src.training.preprocess import (
    build_vocabularies,
    create_playlist_track_map,
    generate_training_pairs,
    load_playlists,
    save_vocab,
)

# ─── hyper-parameters / config ────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR",           "data/")
ARTIFACTS_DIR   = os.getenv("ARTIFACTS_DIR",      "artifacts/")
MLFLOW_URI      = os.environ["MLFLOW_TRACKING_URI"]
MAX_PLAYLISTS   = int(os.getenv("MAX_PLAYLISTS",  "1000"))
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM",  "64"))
OUTPUT_DIM      = int(os.getenv("OUTPUT_DIM",     "64"))
LEARNING_RATE   = float(os.getenv("LEARNING_RATE","1e-3"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE",     "256"))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS",     "10"))
NUM_NEGATIVES   = int(os.getenv("NUM_NEGATIVES",  "4"))
GMM_COMPONENTS  = int(os.getenv("GMM_COMPONENTS", "8"))
SEED            = int(os.getenv("SEED", "42"))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 60)
    playlists = load_playlists(DATA_DIR, MAX_PLAYLISTS)
    track2idx, playlist2idx = build_vocabularies(playlists)
    playlist_track_map = create_playlist_track_map(playlists, track2idx)
    num_tracks = len(track2idx)

    triples = generate_training_pairs(playlist_track_map, num_tracks, NUM_NEGATIVES)
    if not triples:
        raise RuntimeError("No training triples generated. Check data loading.")

    # Save vocab for the API to use
    save_vocab(track2idx, playlist_track_map, ARTIFACTS_DIR)

    # Save playlist_id list for submission.json auto-pick
    valid_pids = sorted(playlist_track_map.keys())
    np.save(os.path.join(ARTIFACTS_DIR, "valid_playlist_ids.npy"), np.array(valid_pids))

    dataset = PlaylistTrackDataset(triples, max_context_len=20)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Building model")
    print("=" * 60)
    model = TwoTowerModel(
        num_tracks=num_tracks,
        embed_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. MLflow run ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training with MLflow tracking")
    print("=" * 60)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("spotify-two-tower")

    with mlflow.start_run(run_name="two-tower-training") as run:
        # Log hyperparameters
        mlflow.log_param("learning_rate",   LEARNING_RATE)
        mlflow.log_param("embedding_dim",   EMBEDDING_DIM)
        mlflow.log_param("output_dim",      OUTPUT_DIM)
        mlflow.log_param("batch_size",      BATCH_SIZE)
        mlflow.log_param("num_epochs",      NUM_EPOCHS)
        mlflow.log_param("num_negatives",   NUM_NEGATIVES)
        mlflow.log_param("num_tracks",      num_tracks)
        mlflow.log_param("num_playlists",   len(playlist_track_map))
        mlflow.log_param("train_samples",   train_size)
        mlflow.log_param("device",          DEVICE)
        mlflow.log_param("seed",            SEED)

        best_val_loss = float("inf")

        for epoch in range(1, NUM_EPOCHS + 1):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            total_loss, total_acc, n_batches = 0.0, 0.0, 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"):
                playlist_tracks  = batch["playlist_tracks"].to(DEVICE)
                playlist_lengths = batch["playlist_length"].to(DEVICE)
                pos_tracks       = batch["pos_track"].to(DEVICE)
                neg_tracks       = batch["neg_track"].to(DEVICE)

                optimizer.zero_grad()
                pl_emb, pos_emb = model(playlist_tracks, playlist_lengths, pos_tracks)
                _,      neg_emb = model(playlist_tracks, playlist_lengths, neg_tracks)

                loss = bpr_loss(pl_emb, pos_emb, neg_emb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_acc  += accuracy(pl_emb, pos_emb, neg_emb)
                n_batches  += 1

            train_loss = total_loss / n_batches
            train_acc  = total_acc  / n_batches

            # ── Validate ───────────────────────────────────────────────────
            model.eval()
            val_loss, val_acc, vn = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    pl_trks = batch["playlist_tracks"].to(DEVICE)
                    pl_lens = batch["playlist_length"].to(DEVICE)
                    pos_t   = batch["pos_track"].to(DEVICE)
                    neg_t   = batch["neg_track"].to(DEVICE)
                    pl_emb, pos_emb = model(pl_trks, pl_lens, pos_t)
                    _,      neg_emb = model(pl_trks, pl_lens, neg_t)
                    val_loss += bpr_loss(pl_emb, pos_emb, neg_emb).item()
                    val_acc  += accuracy(pl_emb, pos_emb, neg_emb)
                    vn       += 1
            val_loss /= max(vn, 1)
            val_acc  /= max(vn, 1)

            scheduler.step()

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc",  train_acc,  step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)
            mlflow.log_metric("val_acc",    val_acc,    step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # ── Log model to MLflow ──────────────────────────────────────────
        print("\nLogging model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="two-tower-recommender",
            registered_model_name="two-tower-recommender",
        )
        mlflow.log_metric("best_val_loss", best_val_loss)
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

    # ── 4. Generate & save track embeddings ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Generating track embeddings")
    print("=" * 60)
    model.eval()
    all_track_ids = torch.arange(1, num_tracks + 1, dtype=torch.long).to(DEVICE)

    batch_size = 512
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_track_ids), batch_size):
            batch_ids = all_track_ids[i : i + batch_size]
            emb = model.encode_track(batch_ids)
            all_embeddings.append(emb.cpu().numpy())

    track_embeddings = np.vstack(all_embeddings)  # (num_tracks, output_dim)
    emb_path = os.path.join(ARTIFACTS_DIR, "track_embeddings.npy")
    np.save(emb_path, track_embeddings)
    print(f"Saved track_embeddings.npy — shape: {track_embeddings.shape}")

    # ── 5. Fit GMM baseline distribution for drift detection ─────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Fitting baseline GMM for drift detection")
    print("=" * 60)
    n_components = min(GMM_COMPONENTS, num_tracks)
    sample_size  = min(5000, len(track_embeddings))
    sample_idx   = np.random.choice(len(track_embeddings), sample_size, replace=False)
    sample_embs  = track_embeddings[sample_idx]

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=200,
        random_state=42,
    )
    gmm.fit(sample_embs)
    dist_path = os.path.join(ARTIFACTS_DIR, "training_track_distribution.pkl")
    with open(dist_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"Saved training_track_distribution.pkl (GMM with {n_components} components)")

    # ── 6. Save run_id for the API to use ─────────────────────────────────────
    run_id_path = os.path.join(ARTIFACTS_DIR, "mlflow_run_id.txt")
    with open(run_id_path, "w") as f:
        f.write(run_id)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"  MLflow run ID:      {run_id}")
    print(f"  Best val loss:      {best_val_loss:.4f}")
    print("=" * 60)

    return run_id


if __name__ == "__main__":
    train()
