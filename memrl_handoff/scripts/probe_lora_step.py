"""One real LoRA optimizer step on a short fixed prompt. Not a research update."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
NAME = "Qwen/Qwen2.5-7B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def load_model(torch, AutoModelForCausalLM):
    kwargs = {"local_files_only": True, "attn_implementation": "sdpa"}
    try:
        return AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, **kwargs)


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    prompt_len = int(input_ids.shape[1])
    completion = tokenizer.encode(" ok", add_special_tokens=False)
    completion_ids = torch.tensor([completion], dtype=torch.long)
    ids = torch.cat([input_ids, completion_ids], dim=1).to("cuda:0")
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    model = load_model(torch, AutoModelForCausalLM).to("cuda:0")
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-6)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 1.0,
    )
    finite = bool(torch.isfinite(grad_norm).item()) and bool(torch.isfinite(loss).item())
    optimizer.step()
    blob = bytearray()
    for name, param in model.named_parameters():
        if param.requires_grad:
            blob.extend(name.encode())
            blob.extend(param.detach().float().cpu().numpy().tobytes())
    report = {
        "model": NAME,
        "loss": float(loss.detach().float().item()),
        "grad_norm": float(grad_norm.detach().float().item()),
        "finite": finite,
        "prompt_tokens": prompt_len,
        "completion_tokens": len(completion),
        "adapter_parameter_sha256": hashlib.sha256(blob).hexdigest(),
        "note": "Single engineering step on one GPU. Not an RL iteration and not a paper metric.",
    }
    path = ROOT / "reports" / "api_probe" / "lora_one_step.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finite": finite, "loss": report["loss"], "grad_norm": report["grad_norm"]}))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
