"""Compare one greedy HF completion with vLLM on the local Qwen checkpoint.

The script records agreement or the exception. It does not fill in a passing result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    "/home/csgrad/utkarshk/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib",
)

ROOT = Path(__file__).resolve().parents[1]
NAME = "Qwen/Qwen2.5-7B-Instruct"


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    input_ids = input_ids.to("cuda:0")
    kwargs = {"local_files_only": True, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, **kwargs)
    model = model.to("cuda:0")
    model.eval()
    generated = model.generate(input_ids, max_new_tokens=8, do_sample=False)
    hf_new = generated[0, input_ids.shape[1]:].tolist()
    del model
    torch.cuda.empty_cache()
    engine = LLM(
        model=NAME, dtype="bfloat16", gpu_memory_utilization=0.5,
        max_model_len=512, enforce_eager=True, disable_log_stats=True,
    )
    params = SamplingParams(temperature=0, max_tokens=8, logprobs=1)
    outputs = engine.generate([prompt_text], params)
    vllm_ids = list(outputs[0].outputs[0].token_ids)
    report = {
        "hf_completion_ids": hf_new,
        "vllm_completion_ids": vllm_ids,
        "token_ids_equal": hf_new == vllm_ids,
        "vllm_version": __import__("vllm").__version__,
    }
    path = ROOT / "reports" / "api_probe" / "vllm_hf.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equal": report["token_ids_equal"], "hf": hf_new, "vllm": vllm_ids}))
    return 0 if report["token_ids_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
