"""GPU test environment: one GPU for in-process tests, offline model, spawn workers, repo cwd."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_TOKEN_FILE", str(REPO / "hf-tok"))

from kmatters.hf_auth import load_hf_token  # noqa: E402

load_hf_token()
