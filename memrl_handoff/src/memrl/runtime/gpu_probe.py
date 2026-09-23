"""Measured Hugging Face probe for the pinned local Qwen checkpoint.

vLLM parity is separate. This module records only values the process computed.
"""

from __future__ import annotations

import json
from pathlib import Path

from memrl.io import atomic_json, sha256_text


def _revision(name: str) -> str | None:
    from pathlib import Path as P
    safe = "models--" + name.replace("/", "--")
    ref = P.home() / ".cache" / "huggingface" / "hub" / safe / "refs" / "main"
    if ref.is_file():
        text = ref.read_text(encoding="utf-8").strip()
        if len(text) == 40:
            return text
    return None


def _max_abs(left, right) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def execute(output: str, devices: str) -> int:
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", devices.split(",")[0])
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memrl.models.logprobs import completion_logprobs

    name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    template = getattr(tokenizer, "chat_template", None) or ""
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = prompt["input_ids"].to("cuda:0")
    prompt_ids = input_ids[0].tolist()
    generated = model.generate(
        input_ids, max_new_tokens=8, do_sample=False,
        return_dict_in_generate=True, output_scores=True,
    )
    new_ids = generated.sequences[0, input_ids.shape[1]:].tolist()
    scored, mode = completion_logprobs(model, prompt_ids, new_ids)
    score_logprobs = []
    for index, token in enumerate(new_ids):
        logp = torch.log_softmax(generated.scores[index][0].float(), dim=-1)[token]
        score_logprobs.append(float(logp.item()))
    teacher = [float(value) for value in scored.tolist()]
    gaps = [abs(a - b) for a, b in zip(teacher, score_logprobs)]
    base_again, _mode = completion_logprobs(model, prompt_ids, new_ids)
    reader_gap = _max_abs(teacher, [float(v) for v in base_again.tolist()])

    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ))
    with torch.no_grad():
        for param_name, param in peft_model.named_parameters():
            if "lora_B" in param_name:
                param.normal_(0, 0.05)
    adapted, _ = completion_logprobs(peft_model, prompt_ids, new_ids)
    adapted_values = [float(v) for v in adapted.tolist()]
    with peft_model.disable_adapter():
        disabled, _ = completion_logprobs(peft_model, prompt_ids, new_ids)
    disabled_values = [float(v) for v in disabled.tolist()]
    report = {
        "model": name,
        "revision_from_cache_ref": _revision(name),
        "local_files_only": True,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "chat_template_sha256": sha256_text(template),
        "prompt_tokens": len(prompt_ids),
        "completion_ids": new_ids,
        "logprob_mode": mode,
        "mae": sum(gaps) / len(gaps),
        "max_abs": max(gaps),
        "reader_repeat_max_abs": reader_gap,
        "adapter_disabled_max_abs_vs_base": _max_abs(teacher, disabled_values),
        "adapter_enabled_max_abs_vs_base": _max_abs(teacher, adapted_values),
        "reader_adapter": None,
        "peft_version": __import__("peft").__version__,
        "vllm": "not_run",
        "note": "HF greedy scores versus teacher forcing on one GPU. Not a training result.",
    }
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    atomic_json(path / "hf_logprob.json", report)
    print(json.dumps({
        "mae": report["mae"],
        "max_abs": report["max_abs"],
        "adapter_delta": report["adapter_enabled_max_abs_vs_base"],
        "reader_delta": report["adapter_disabled_max_abs_vs_base"],
        "revision": report["revision_from_cache_ref"],
    }))
    ok = (
        report["mae"] <= 0.02
        and report["max_abs"] <= 0.1
        and report["adapter_disabled_max_abs_vs_base"] <= 0.02
        and report["adapter_enabled_max_abs_vs_base"] > 0.0
    )
    return 0 if ok else 1
