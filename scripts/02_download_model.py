"""Download the pinned model revision and write configs/model_lock.yaml (PLAN.md §5.4)."""
from kmatters.hf_auth import load_hf_token
tok = load_hf_token()
import datetime, hashlib, pathlib, yaml
from huggingface_hub import HfApi, snapshot_download
REPO = "Qwen/Qwen3-4B-Instruct-2507"
sha = HfApi(token=tok).model_info(REPO).sha
path = snapshot_download(REPO, revision=sha, local_dir="models/Qwen3-4B-Instruct-2507", token=tok)
config_sha256 = hashlib.sha256(pathlib.Path(path, "config.json").read_bytes()).hexdigest()
yaml.safe_dump({"repo": REPO, "revision": sha, "local_dir": str(path),
                "config_sha256": config_sha256,
                "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
               open("configs/model_lock.yaml", "w"))
print(f"downloaded {REPO}@{sha} to {path}")
