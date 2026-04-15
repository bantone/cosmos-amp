"""
Model Pre-download Script
=========================
Downloads the Cosmos Reason2-8B model weights to the local HuggingFace cache
so the application starts immediately without fetching 16 GB at launch time.

Run this as a Cloudera AI Workbench Job before starting the application.
No GPU required — CPU-only with ~8 GB RAM is sufficient.
"""

import os
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-8B")
CACHE_DIR  = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"Model   : {MODEL_NAME}")
print(f"Cache   : {CACHE_DIR}")
print()

# huggingface_hub is a required dependency of transformers — always available.
# snapshot_download fetches all model files to the local cache without loading
# weights into memory, so this job needs no GPU and minimal RAM.
from huggingface_hub import snapshot_download

print("Starting download — this may take a while (~16 GB)...")
snapshot_download(
    repo_id=MODEL_NAME,
    cache_dir=os.path.join(CACHE_DIR, "hub"),
)
print()
print("Download complete. Model is ready for the application.")
