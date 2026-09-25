"""HF model + PEFT LoRA + AdamW + adapter I/O (PLAN.md §8.9).

The loss is Appendix A.3's L = -(1/T_norm) sum_c w_c sum_t log pi, built from the verbatim
`pack_microbatches` and `completion_logprob_sums` so that per-token HF log-probs (for the
HF-vs-vLLM diagnostics) come from the same forward pass as the gradient.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from .rl.logprobs import Call, completion_logprob_sums, flat_grad, pack_microbatches, trainable_params
from .seeding import seed_from


def _sha256_tensors(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().to("cpu").contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def adapter_sha256_from_dir(path) -> str:
    """sha256 over the LoRA tensors' bytes in sorted key order (not over the file)."""
    return _sha256_tensors(load_file(str(Path(path) / "adapter_model.safetensors")))


class Trainer:
    def __init__(self, cfg, pad_id: int, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """dtype is bf16 everywhere except the fp32 correctness checks of T2.8/T2.9."""
        self.cfg, self.pad_id, self.device = cfg, pad_id, device
        # Variable-size microbatches fragment the caching allocator (reserved ~2x the peak in use);
        # expandable segments keep reserved close to the peak. Set in this process only, after the
        # vLLM engine process has started with its own allocator settings.
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.manual_seed(seed_from(cfg.train.seed, "torch"))
        base = AutoModelForCausalLM.from_pretrained(cfg.model.local_dir, dtype=dtype,
                                                    attn_implementation=cfg.model.attn_implementation).to(device)
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(cfg.train.seed + cfg.lora.init_seed_offset)
        lc = cfg.lora
        self.model = get_peft_model(base, LoraConfig(r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
                                                     bias="none", target_modules=list(lc.target_modules),
                                                     task_type="CAUSAL_LM"))
        self.model.train()          # gradient checkpointing is active only in train mode; no dropout anywhere
        self.params = trainable_params(self.model)
        # PEFT keeps adapter weights in fp32 for a bf16 base (autocast_adapter_dtype=True)
        assert self.params and all(p.dtype == torch.float32 for p in self.params), "LoRA params must be fp32"
        self.n_trainable = sum(p.numel() for p in self.params)
        self.reset_optimizer()

    def reset_optimizer(self):
        t = self.cfg.train
        self.opt = torch.optim.AdamW(self.params, lr=t.lr, betas=tuple(t.betas), eps=t.eps,
                                     weight_decay=t.weight_decay, fused=True)

    # ------------------------------------------------------------------ loss
    def _accumulate(self, calls, T_norm, max_tokens):
        """Backward of -(1/T_norm) sum_c w_c sum_t log pi over microbatches (grads accumulate).
        Same computation as rl.logprobs.accumulate_policy_gradient, plus per-token log-probs."""
        lengths = [len(c.prompt_ids) + len(c.completion_ids) for c in calls]
        total, lp_sums, tok_lps = 0.0, [None] * len(calls), [None] * len(calls)
        for mb in pack_microbatches(lengths, max_tokens):
            sub = [calls[i] for i in mb]
            s, tok_lp, _ = completion_logprob_sums(self.model, sub, self.pad_id, self.device)
            w = torch.tensor([c.weight for c in sub], device=self.device, dtype=s.dtype)
            loss = -(w * s).sum() / T_norm
            loss.backward()
            total += float(loss.detach())
            tl = tok_lp.detach().float().cpu()
            off = 0
            for j, i in enumerate(mb):
                C = len(calls[i].completion_ids)
                tok_lps[i] = tl[off:off + C]
                lp_sums[i] = float(s[j].detach())
                off += C
        return total, lp_sums, tok_lps

    def grad_norm(self) -> float:
        norms = [p.grad.detach().float().norm() for p in self.params if p.grad is not None]
        return float(torch.stack(norms).norm()) if norms else 0.0

    def step(self, calls, T_norm, max_tokens, lr=None, roles=None) -> dict:
        """One AdamW step on the weighted policy-gradient loss. Prefix and future calls are
        packed and timed separately. lr=0.0 in profiling; None in training. An lr override
        applies to this step only."""
        saved_lr = [g["lr"] for g in self.opt.param_groups]
        try:
            if lr is not None:
                for g in self.opt.param_groups:
                    g["lr"] = lr
            return self._step(calls, T_norm, max_tokens, roles)
        finally:
            for g, v in zip(self.opt.param_groups, saved_lr):
                g["lr"] = v

    def _step(self, calls, T_norm, max_tokens, roles) -> dict:
        self.opt.zero_grad(set_to_none=True)
        roles = roles or ["all"] * len(calls)
        groups = {}
        for idx, r in enumerate(roles):
            groups.setdefault(r, []).append(idx)
        loss, lp_sums, tok_lps, times = 0.0, [None] * len(calls), [None] * len(calls), {}
        for role, idxs in groups.items():
            torch.cuda.synchronize()
            t0 = perf_counter()
            l, s, tl = self._accumulate([calls[i] for i in idxs], T_norm, max_tokens)
            torch.cuda.synchronize()
            times[role] = perf_counter() - t0
            loss += l
            for j, i in enumerate(idxs):
                lp_sums[i], tok_lps[i] = s[j], tl[j]
        t0 = perf_counter()
        gn = self.grad_norm()
        if not (np.isfinite(gn) and np.isfinite(loss)):
            self.opt.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite loss/grad norm (loss={loss}, grad_norm={gn}); step skipped")
        clip = self.cfg.train.grad_clip
        clipped = bool(clip is not None and gn > clip)
        if clipped:
            torch.nn.utils.clip_grad_norm_(self.params, clip)
        self.opt.step()
        torch.cuda.synchronize()
        t_opt = perf_counter() - t0
        return {"loss": loss, "grad_norm": gn, "clipped": clipped,
                "t_train_prefix": times.get("prefix", 0.0), "t_train_future": times.get("future", 0.0),
                "t_train_groups": times, "t_opt": t_opt, "lp_sums": lp_sums, "tok_lps": tok_lps}

    def score_grad(self, pairs, T_norm, max_tokens) -> torch.Tensor:
        """Flat fp32 gradient of (1/T_norm) sum_calls sum_t log pi over (prompt_ids, completion_ids)
        pairs (the variance tool's score vector). Leaves .grad cleared."""
        self.opt.zero_grad(set_to_none=True)
        self._accumulate([Call(p, c, 1.0) for p, c in pairs], T_norm, max_tokens)
        g = -flat_grad(self.model)          # the loss is -(1/T_norm) sum log pi
        self.opt.zero_grad(set_to_none=True)
        return g

    # ------------------------------------------------------------------ adapter / state I/O
    def adapter_state(self) -> dict:
        return get_peft_model_state_dict(self.model)

    def adapter_sha256(self) -> str:
        return _sha256_tensors(self.adapter_state())

    def save_adapter(self, path):
        self.model.save_pretrained(str(path), safe_serialization=True)

    def load_adapter_weights(self, path):
        sd = load_file(str(Path(path) / "adapter_model.safetensors"), device=self.device)
        res = set_peft_model_state_dict(self.model, sd)
        missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        if missing or getattr(res, "unexpected_keys", []):
            raise RuntimeError(f"adapter load mismatch: missing={missing[:3]} unexpected={res.unexpected_keys[:3]}")

    def save_state(self, path):
        torch.save({"opt": self.opt.state_dict(),
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                            "numpy": np.random.get_state(), "python": random.getstate()}}, str(path))

    def load_state(self, adapter_path, state_path):
        self.load_adapter_weights(adapter_path)
        st = torch.load(str(state_path), map_location="cpu", weights_only=False)
        self.opt.load_state_dict(st["opt"])
        rng = st["rng"]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    def optimizer_step_count(self) -> int:
        steps = [int(s["step"]) for s in self.opt.state.values() if "step" in s]
        return max(steps) if steps else 0
