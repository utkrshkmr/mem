# kmatters/hf_auth.py
import logging, os, stat
from pathlib import Path

_TOKEN = None

def _candidates():
    env = os.environ.get("HF_TOKEN_FILE")
    if env:
        yield Path(env).expanduser()
    yield Path.cwd() / "hf-tok"
    yield Path.home() / "hf-tok"

def load_hf_token(set_env: bool = True) -> str:
    """Read the token from hf-tok, export HF_TOKEN for huggingface_hub and vLLM, and return it.
    Call once, at the top of every entry-point script, before importing vllm/transformers."""
    global _TOKEN
    for p in _candidates():
        if p.is_file():
            tok = p.read_text().strip()
            if not tok:
                raise ValueError(f"{p} is empty")
            if not tok.startswith("hf_"):
                logging.warning("hf-tok content does not start with 'hf_'; using it anyway")
            mode = p.stat().st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH):
                logging.warning("%s is readable by group/others; run: chmod 600 %s", p, p)
            _TOKEN = tok
            if set_env:
                os.environ["HF_TOKEN"] = tok          # read by huggingface_hub and vLLM subprocesses
                os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            return tok
    raise FileNotFoundError("Hugging Face token file not found. Put your token in ./hf-tok "
                            "(one line, chmod 600) or set HF_TOKEN_FILE.")

class RedactFilter(logging.Filter):
    def filter(self, record):
        if _TOKEN and _TOKEN in str(record.getMessage()):
            record.msg = str(record.getMessage()).replace(_TOKEN, "hf_***")
            record.args = ()
        return True
