#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_training.sh  —  trains the model locally (outside Docker)
#
# Prerequisites:
#   pip install -r requirements.txt
#   (Optional) Place Spotify MPD JSON files in data/
#              If no files exist, synthetic data will be generated automatically.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export DATA_DIR="${DATA_DIR:-data}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
export MAX_PLAYLISTS="${MAX_PLAYLISTS:-1000}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
export OUTPUT_DIM="${OUTPUT_DIM:-64}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"

echo "=================================================="
echo "  Spotify Two-Tower Recommender — Training"
echo "=================================================="
echo "  MLFLOW_TRACKING_URI : $MLFLOW_TRACKING_URI"
echo "  DATA_DIR            : $DATA_DIR"
echo "  ARTIFACTS_DIR       : $ARTIFACTS_DIR"
echo "  MAX_PLAYLISTS       : $MAX_PLAYLISTS"
echo "  NUM_EPOCHS          : $NUM_EPOCHS"
echo "=================================================="

mkdir -p "$DATA_DIR" "$ARTIFACTS_DIR"

python src/training/train.py

echo ""
echo "Training complete. Generating submission.json..."
python scripts/generate_submission.py

echo ""
echo "Done! Next steps:"
echo "  1. docker-compose up -d --build"
echo "  2. curl -X POST http://localhost:8000/recommend \\"
echo '       -H "Content-Type: application/json" \\'
echo '       -d "{\"playlist_id\": $(cat submission.json | python -c \"import sys,json; print(json.load(sys.stdin)['"'"'test_playlist_id'"'"'])\"), \"top_k\": 5}"'
