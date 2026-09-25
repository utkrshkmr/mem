"""T0.1-T0.6: node, versions, HF auth, vLLM API surface, model lock, chat template (Gate G0)."""
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models" / "Qwen3-4B-Instruct-2507"


def _env(**extra):
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    env.update(extra)
    return env


def test_T0_1_node_check(record):
    p = subprocess.run(["bash", str(REPO / "scripts" / "00_check_node.sh")], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    node = json.loads((REPO / "reports" / "node.json").read_text())
    assert node["passed"] and len(node["gpus"]) == 4
    record("T0.1", driver=node["driver_version"], install_path=node["install_path"],
           gpus=[g["name"] for g in node["gpus"]])


def test_T0_2_versions_and_device_count(record):
    import peft
    import torch
    import transformers
    import vllm
    env = _env()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    n = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                       env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert n == "4", n
    record("T0.2", torch=torch.__version__, torch_cuda=torch.version.cuda, vllm=vllm.__version__,
           transformers=transformers.__version__, peft=peft.__version__,
           python=sys.version.split()[0], device_count_unrestricted=int(n))


def test_T0_3_hf_auth_and_no_token_leak(record):
    from kmatters.hf_auth import load_hf_token
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        tok = load_hf_token()
    finally:
        os.chdir(cwd)
    from huggingface_hub import HfApi
    who = HfApi(token=tok).whoami()
    assert who.get("name"), "whoami returned no user"
    # grep for the token with a pattern file, so it never appears on a command line
    with tempfile.TemporaryDirectory() as d:
        pat = Path(d) / "pat"
        pat.touch(mode=0o600)
        pat.write_text(tok + "\n")
        p = subprocess.run(["grep", "-rlF", "-f", str(pat), "--exclude=hf-tok", "."], cwd=REPO,
                           capture_output=True, text=True)
    assert p.returncode == 1, f"token found in: {p.stdout.split()}" if p.returncode == 0 else p.stderr
    record("T0.3", whoami_ok=True, token_found_in=[])


def test_T0_4_vllm_api_surface(record):
    import dataclasses
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt  # noqa: F401
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    def params(f):
        return set(inspect.signature(f).parameters)

    checks = {
        "AsyncLLM.from_engine_args": callable(getattr(AsyncLLM, "from_engine_args", None)),
        "AsyncLLM.generate(lora_request, request_id)": {"lora_request", "request_id"} <= params(AsyncLLM.generate),
        "AsyncLLM.add_lora": callable(getattr(AsyncLLM, "add_lora", None)),
        "AsyncLLM.remove_lora": callable(getattr(AsyncLLM, "remove_lora", None)),
        "LoRARequest(lora_name, lora_int_id, lora_path)":
            list(inspect.signature(LoRARequest).parameters)[:3] == ["lora_name", "lora_int_id", "lora_path"],
        "SamplingParams(logprobs, prompt_logprobs, output_kind, seed)":
            {"logprobs", "prompt_logprobs", "output_kind", "seed"} <= params(SamplingParams),
        "AsyncEngineArgs fields": {"kv_cache_memory_bytes", "enable_lora", "max_loras", "max_lora_rank",
                                   "max_cpu_loras", "max_num_seqs", "max_num_batched_tokens",
                                   "enable_prefix_caching"} <= {f.name for f in dataclasses.fields(AsyncEngineArgs)},
        "RequestOutputKind.FINAL_ONLY": hasattr(RequestOutputKind, "FINAL_ONLY"),
    }
    record("T0.4", checks=checks)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}


def test_T0_5_model_lock_and_offline_load(record):
    lock = yaml.safe_load((REPO / "configs" / "model_lock.yaml").read_text())
    assert {"repo", "revision", "local_dir", "config_sha256", "downloaded_at"} <= set(lock)
    assert hashlib.sha256((MODEL_DIR / "config.json").read_bytes()).hexdigest() == lock["config_sha256"]

    hf = ("import torch; from transformers import AutoModelForCausalLM, AutoTokenizer\n"
          f"p = {str(MODEL_DIR)!r}\n"
          "tok = AutoTokenizer.from_pretrained(p)\n"
          "m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()\n"
          "ids = torch.tensor([tok('Hello')['input_ids']]).cuda()\n"
          "with torch.no_grad(): out = m(input_ids=ids).logits\n"
          "assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()\n"
          "print('HF_OK', m.config.num_hidden_layers)\n")
    p = subprocess.run([sys.executable, "-c", hf], env=_env(CUDA_VISIBLE_DEVICES="0"),
                       capture_output=True, text=True, timeout=900)
    assert "HF_OK 36" in p.stdout, p.stderr[-3000:]

    vl = ("from vllm import LLM, SamplingParams\n"
          "from vllm.inputs import TokensPrompt\n"
          "if __name__ == '__main__':\n"
          f"    llm = LLM(model={str(MODEL_DIR)!r}, dtype='bfloat16', max_model_len=1024,\n"
          "              kv_cache_memory_bytes=4 * 2**30, enforce_eager=True, seed=0)\n"
          "    tok = llm.get_tokenizer()\n"
          "    ids = tok.apply_chat_template([{'role': 'user', 'content': 'Say hi.'}],\n"
          "                                  add_generation_prompt=True, tokenize=True, return_dict=False)\n"
          "    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=8, temperature=0))\n"
          "    assert len(out[0].outputs[0].token_ids) > 0\n"
          "    print('VLLM_OK', repr(out[0].outputs[0].text))\n")
    p2 = subprocess.run([sys.executable, "-c", vl], env=_env(CUDA_VISIBLE_DEVICES="0"),
                        capture_output=True, text=True, timeout=1200)
    assert "VLLM_OK" in p2.stdout, p2.stderr[-4000:]
    record("T0.5", revision=lock["revision"], config_sha256=lock["config_sha256"],
           hf_offline_load=True, vllm_offline_load=True,
           vllm_sample=p2.stdout.strip().splitlines()[-1])


def test_T0_6_chat_template(record):
    from transformers import AutoTokenizer
    from kmatters.prompts import EMPTY_MEMORY, WRITER_SYSTEM, writer_ids
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    msgs = [{"role": "system", "content": "SYSTEM TEXT"}, {"role": "user", "content": "hello"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    assert isinstance(ids, list) and all(type(t) is int for t in ids)
    s = tok.decode(ids)
    assert "<|im_start|>system" in s and "SYSTEM TEXT" in s and s.endswith("<|im_start|>assistant\n")
    w = tok.decode(writer_ids(tok, EMPTY_MEMORY, "T01: A paid B $1 for rent.", 512))
    assert WRITER_SYSTEM.format(B=512) in w and w.endswith("<|im_start|>assistant\n")
    assert "<think>" not in w
    record("T0.6", n_ids=len(ids), tail=s[-30:])
