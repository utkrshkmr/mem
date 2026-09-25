"""Token log-probabilities of completions and the weighted policy-gradient loss.

Design rules (see PLAN.md section 8.9):
  * Train on the exact token ids vLLM returned (prompt_token_ids + token_ids).
  * Right-pad, run the decoder once, gather hidden states only at positions
    that predict completion tokens, apply lm_head in chunks, log_softmax in fp32.
  * One optimizer step per iteration; grads accumulate across microbatches.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Call:
    prompt_ids: list
    completion_ids: list
    weight: float = 0.0          # coefficient w_c from rl/estimator.call_weights


def pack_microbatches(lengths, max_tokens):
    """Greedy packing of calls (sorted by length) so padded_len * n <= max_tokens.
    Returns list of index lists. A single call longer than max_tokens gets its own
    microbatch."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and new_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _decoder_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def completion_logprob_sums(model, calls, pad_id, device, head_chunk=4096):
    """Returns a 1-D tensor: for each call, sum over completion tokens of log pi.
    Differentiable w.r.t. trainable params."""
    decoder, head = _decoder_and_head(model)
    seqs = [c.prompt_ids + c.completion_ids for c in calls]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, :len(s)] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = torch.arange(T, device=device).unsqueeze(0).expand(len(seqs), T)
    hidden = decoder(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    # positions p predict token p+1; completion tokens sit at [P, P+C)
    rows, cols, targets, owner = [], [], [], []
    for b, c in enumerate(calls):
        P, C = len(c.prompt_ids), len(c.completion_ids)
        rows += [b] * C
        cols += list(range(P - 1, P + C - 1))
        targets += c.completion_ids
        owner += [b] * C
    rows = torch.tensor(rows, device=device)
    cols = torch.tensor(cols, device=device)
    targets = torch.tensor(targets, device=device)
    owner = torch.tensor(owner, device=device)
    h = hidden[rows, cols]                                # [n_tok, H]
    tok_lp = []
    for s in range(0, h.shape[0], head_chunk):
        logits = head(h[s:s + head_chunk]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp.append(lp.gather(1, targets[s:s + head_chunk, None]).squeeze(1))
    tok_lp = torch.cat(tok_lp)
    out = torch.zeros(len(calls), device=device, dtype=tok_lp.dtype)
    return out.index_add(0, owner, tok_lp), tok_lp, owner


def accumulate_policy_gradient(model, calls, T_norm, pad_id, device, max_tokens):
    """Backward of L = -(1/T_norm) * sum_c w_c * sum_t log pi, microbatched.
    Grads accumulate in .grad; caller zeroes grads before and steps after.
    Returns the scalar loss value (float) and per-call logprob sums (list)."""
    lengths = [len(c.prompt_ids) + len(c.completion_ids) for c in calls]
    total, lp_sums = 0.0, [None] * len(calls)
    for mb in pack_microbatches(lengths, max_tokens):
        sub = [calls[i] for i in mb]
        s, _, _ = completion_logprob_sums(model, sub, pad_id, device)
        w = torch.tensor([c.weight for c in sub], device=device, dtype=s.dtype)
        loss = -(w * s).sum() / T_norm
        loss.backward()
        total += float(loss.detach())
        for j, i in enumerate(mb):
            lp_sums[i] = float(s[j].detach())
    return total, lp_sums


def trainable_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad]


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten().float() if p.grad is not None
                      else torch.zeros(p.numel(), device=p.device) for p in trainable_params(model)])
