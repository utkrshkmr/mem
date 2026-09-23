"""Two-rank DDP gradient versus one rank, on Qwen2.5-0.5B. Not the 7B study run."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def _load(torch, AutoModelForCausalLM):
    kwargs = {"local_files_only": True, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, **kwargs)
    return model


def _batch(tokenizer, torch):
    messages = [{"role": "user", "content": "Reply with the single word ok."}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    prompt_len = int(input_ids.shape[1])
    completion = torch.tensor([tokenizer.encode(" ok", add_special_tokens=False)], dtype=torch.long)
    ids = torch.cat([input_ids, completion], dim=1)
    labels = ids.clone()
    labels[:, :prompt_len] = -100
    return ids, labels


def _peft(model):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    ))


def _grad_sample(model, torch):
    parts = []
    for _name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            parts.append(param.grad.detach().float().cpu().reshape(-1)[:32])
    if not parts:
        raise RuntimeError("no gradients")
    return torch.cat(parts)


def single(state_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
    ids, labels = _batch(tokenizer, torch)
    model = _peft(_load(torch, AutoModelForCausalLM))
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model = model.to("cuda:0")
    model.train()
    loss = model(input_ids=ids.to("cuda:0"), labels=labels.to("cuda:0")).loss
    loss.backward()
    return _grad_sample(model, torch)


def _worker(rank: int, world: int, result_path: str, state_path: str) -> None:
    import os
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(seconds=120))
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(NAME, local_files_only=True)
    ids, labels = _batch(tokenizer, torch)
    model = _peft(_load(torch, AutoModelForCausalLM))
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model = model.to(f"cuda:{rank}")
    model.train()
    ddp = DDP(model, device_ids=[rank])
    loss = ddp(input_ids=ids.to(f"cuda:{rank}"), labels=labels.to(f"cuda:{rank}")).loss
    loss.backward()
    if rank == 0:
        torch.save(_grad_sample(model, torch), result_path)
    dist.destroy_process_group()


def main() -> int:
    import torch
    import torch.multiprocessing as mp
    from transformers import AutoModelForCausalLM
    state_path = ROOT / "reports" / "api_probe" / "ddp_init.pt"
    result = ROOT / "reports" / "api_probe" / "ddp_grad.pt"
    result.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    initial = _peft(_load(torch, AutoModelForCausalLM))
    torch.save(initial.state_dict(), state_path)
    del initial
    one = single(str(state_path))
    mp.spawn(_worker, args=(2, str(result), str(state_path)), nprocs=2, join=True)
    two = torch.load(result, map_location="cpu", weights_only=True)
    gap = float((one - two).abs().max().item())
    report = {
        "model": NAME,
        "max_abs_grad_sample": gap,
        "equal_within_1e-4": gap <= 1e-4,
        "note": "0.5B engineering check on GPUs 2 and 3. Both ranks see the same batch, so DDP's average matches the single-process gradient up to numerics. This is not the 7B study run.",
    }
    (ROOT / "reports" / "api_probe" / "ddp_grad.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"max_abs_grad_sample": gap, "ok": report["equal_within_1e-4"]}))
    return 0 if report["equal_within_1e-4"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
