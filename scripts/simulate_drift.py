"""
Drift simulation script.

Sends requests to /recommend with playlist IDs cycling through the dataset
quickly enough to fill the drift-detection window (default: 100 requests)
within 60 seconds, triggering the KL divergence alert.

Usage:
    python scripts/simulate_drift.py [--api-url URL] [--playlist-id ID] [--count N]
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np


def load_valid_ids(artifacts_dir: str = "artifacts") -> list[int]:
    """Load valid playlist IDs from artifacts directory."""
    path = os.path.join(artifacts_dir, "valid_playlist_ids.npy")
    if os.path.exists(path):
        return np.load(path).tolist()
    # Fall back to a range if not found
    return list(range(1000))


def simulate_drift(
    api_url: str = "http://localhost:8000",
    artifacts_dir: str = "artifacts",
    num_requests: int = 150,
    top_k: int = 10,
):
    valid_ids = load_valid_ids(artifacts_dir)
    if not valid_ids:
        print("[ERROR] No valid playlist IDs found. Run training first.")
        sys.exit(1)

    print(f"Simulating drift: sending {num_requests} requests to {api_url}/recommend")
    print(f"Using {len(valid_ids)} valid playlist IDs")
    print("This will trigger drift detection when the window fills up...")
    print("-" * 60)

    success, failed = 0, 0
    start_time = time.time()

    with httpx.Client(timeout=30.0) as client:
        for i in range(num_requests):
            pid = random.choice(valid_ids)
            payload = {"playlist_id": int(pid), "top_k": top_k}

            try:
                resp = client.post(f"{api_url}/recommend", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    cache_status = "HIT " if data["retrieved_from_cache"] else "MISS"
                    print(
                        f"[{i+1:3d}/{num_requests}] pid={pid:<6} "
                        f"cache={cache_status} "
                        f"tracks={data['recommended_track_ids'][:3]}..."
                    )
                    success += 1
                else:
                    print(f"[{i+1:3d}/{num_requests}] pid={pid} → HTTP {resp.status_code}")
                    failed += 1
            except Exception as e:
                print(f"[{i+1:3d}/{num_requests}] pid={pid} → ERROR: {e}")
                failed += 1

            # Small sleep to avoid hammering, but fast enough to finish <60s
            time.sleep(0.1)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Done in {elapsed:.1f}s — success={success} failed={failed}")
    print("\nCheck the api service logs for the DRIFT DETECTED message:")
    print("  docker-compose logs api | grep DRIFT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate concept drift")
    parser.add_argument("--api-url",      default="http://localhost:8000")
    parser.add_argument("--artifacts-dir",default="artifacts")
    parser.add_argument("--count",        type=int, default=150)
    parser.add_argument("--top-k",        type=int, default=10)
    args = parser.parse_args()

    simulate_drift(
        api_url=args.api_url,
        artifacts_dir=args.artifacts_dir,
        num_requests=args.count,
        top_k=args.top_k,
    )
