"""CPU tests of the loss code on a tiny randomly initialized Qwen3 + LoRA."""
import pytest
import torch

from kmatters.rl.logprobs import (Call, accumulate_policy_gradient, completion_logprob_sums,
                                  flat_grad, pack_microbatches)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=500, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    base = Qwen3ForCausalLM(cfg).float()
    lcfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(base, lcfg)
    # make LoRA B non-zero so the adapter matters
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)
    return m


def _calls(seed, n=13):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        P = int(torch.randint(3, 25, (1,), generator=g))
        C = int(torch.randint(1, 12, (1,), generator=g))
        out.append(Call(torch.randint(1, 500, (P,), generator=g).tolist(),
                        torch.randint(1, 500, (C,), generator=g).tolist(),
                        float(torch.randn(1, generator=g))))
    return out


def _naive_sums(model, calls):
    res = []
    for c in calls:
        ids = torch.tensor([c.prompt_ids + c.completion_ids])
        logits = model(input_ids=ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        P = len(c.prompt_ids)
        res.append(sum(lp[P - 1 + t, tok] for t, tok in enumerate(c.completion_ids)))
    return torch.stack(res)


def test_logprob_sums_match_naive(tiny):
    calls = _calls(1)
    s, _, _ = completion_logprob_sums(tiny, calls, pad_id=0, device="cpu", head_chunk=7)
    ref = _naive_sums(tiny, calls)
    assert torch.allclose(s, ref, atol=1e-4), (s - ref).abs().max()


def test_microbatch_partition_invariance(tiny):
    calls = _calls(2, n=17)
    grads = []
    for max_tokens in (10_000, 60, 25):
        tiny.zero_grad(set_to_none=True)
        accumulate_policy_gradient(tiny, calls, T_norm=64.0, pad_id=0,
                                   device="cpu", max_tokens=max_tokens)
        grads.append(flat_grad(tiny).clone())
    for g in grads[1:]:
        rel = (g - grads[0]).norm() / grads[0].norm()
        assert rel < 1e-5, rel


def test_gradient_matches_naive_objective(tiny):
    calls = _calls(3, n=9)
    tiny.zero_grad(set_to_none=True)
    accumulate_policy_gradient(tiny, calls, T_norm=10.0, pad_id=0, device="cpu", max_tokens=40)
    g1 = flat_grad(tiny).clone()
    tiny.zero_grad(set_to_none=True)
    w = torch.tensor([c.weight for c in calls])
    (-(w * _naive_sums(tiny, calls)).sum() / 10.0).backward()
    g2 = flat_grad(tiny).clone()
    assert (g1 - g2).norm() / g2.norm() < 1e-4


def test_only_lora_trainable_and_dropout_zero(tiny):
    names = [n for n, p in tiny.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    for mod in tiny.modules():
        if hasattr(mod, "lora_dropout"):
            for d in mod.lora_dropout.values():
                assert isinstance(d, torch.nn.Identity)


def test_packing_respects_budget():
    lens = [5, 50, 7, 30, 12, 12, 3, 80]
    mbs = pack_microbatches(lens, 60)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(lens)))
    for mb in mbs:
        assert len(mb) == 1 or max(lens[i] for i in mb) * len(mb) <= 60


def test_sign_of_update(tiny):
    """A positive weight must increase the call's log-prob after one SGD step; negative must decrease."""
    import copy
    for sign in (+1.0, -1.0):
        m = copy.deepcopy(tiny)
        c = _calls(5, n=1)[0]
        c.weight = sign
        before = float(completion_logprob_sums(m, [c], 0, "cpu")[0][0].detach())
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        opt.zero_grad()
        accumulate_policy_gradient(m, [c], T_norm=1.0, pad_id=0, device="cpu", max_tokens=10_000)
        opt.step()
        after = float(completion_logprob_sums(m, [c], 0, "cpu")[0][0].detach())
        assert (after - before) * sign > 0, (sign, before, after)
