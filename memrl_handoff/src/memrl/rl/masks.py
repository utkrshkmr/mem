"""Completion log-probability masking. Prompt and padding positions are excluded."""

from __future__ import annotations


def completion_mask(prompt_length: int, completion_length: int, sequence_length: int) -> list[bool]:
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("nonempty prompt and completion required")
    if sequence_length < prompt_length + completion_length:
        raise ValueError("sequence shorter than prompt plus completion")
    mask = [False] * sequence_length
    # Positions that predict completion tokens are prompt_length-1 .. prompt_length+completion_length-2
    # The mask is over target positions, not logit positions.
    for index in range(prompt_length, prompt_length + completion_length):
        mask[index] = True
    return mask


def masked_logprob_sum(token_logprobs: list[float], mask: list[bool]) -> float:
    if len(token_logprobs) != len(mask):
        raise ValueError("logprob mask length mismatch")
    if not any(mask):
        raise ValueError("completion mask is empty")
    total = 0.0
    for value, keep in zip(token_logprobs, mask):
        if keep:
            total += value
    return total
