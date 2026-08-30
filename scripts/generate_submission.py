"""
Generate submission.json after training.
Picks a valid test_playlist_id from the training data.

Usage:
    python scripts/generate_submission.py
"""

import json
import os
import numpy as np

ARTIFACTS_DIR   = os.getenv("ARTIFACTS_DIR",    "artifacts")
MLFLOW_PORT     = os.getenv("MLFLOW_PORT",       "5001")
MLFLOW_HOST     = os.getenv("MLFLOW_HOST",       "localhost")

def main():
    ids_path = os.path.join(ARTIFACTS_DIR, "valid_playlist_ids.npy")
    if os.path.exists(ids_path):
        ids = np.load(ids_path).tolist()
        test_id = int(ids[0]) if ids else 0
    else:
        test_id = 0
        print("[WARNING] valid_playlist_ids.npy not found. Using pid=0.")

    submission = {
        "test_playlist_id": test_id,
        "mlflow_server_url": f"http://{MLFLOW_HOST}:{MLFLOW_PORT}",
    }

    with open("submission.json", "w") as f:
        json.dump(submission, f, indent=2)

    print(f"submission.json written:")
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
