"""Exact scalar reference for the branched REINFORCE/RLOO loss.

Layout: rewards[h][g][k], H histories, G >= 2 independent policy samples,
K environment continuations per prefix. For a given (h, k), all g see the same
exogenous suffix; their policy randomness MUST remain independent. Suffixes are
not future information in the prefix policy input. Baselines/rewards are stopped
gradients. Correlated writer sampling across g invalidates this baseline proof.

    A[h,g,k] = R[h,g,k] - mean_{j != g} R[h,j,k]
    Abar[h,g] = mean_k A[h,g,k]
    L = -1/(H*G*Z) * sum_{h,g} (
          Abar[h,g] * sum_prefix_calls log p
          + mean_k A[h,g,k] * sum_suffix_calls[k] log p)

Each physical prefix call appears once. Z is a FIXED scale, not sampled token
count. Call subsampling multiplies each selected call by 1 / its known inclusion
probability. The code returns scalar loss coefficients for production tensor
code to reproduce; it does not supply gradients or run model training itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Sequence

from .contracts import finite_number, strict_integer, valid_text

RewardCube = tuple[tuple[tuple[float, ...], ...], ...]


def validate_rewards(rewards: Sequence[Sequence[Sequence[float]]]) -> RewardCube:
    if not rewards:
        raise ValueError("H must be >= 1")
    group_size = len(rewards[0])
    if group_size < 2:
        raise ValueError("G must be >= 2 for leave-one-out advantages")
    branch_count = len(rewards[0][0])
    if branch_count < 1:
        raise ValueError("K must be >= 1")
    frozen = []
    for group in rewards:
        if len(group) != group_size:
            raise ValueError("ragged G dimension")
        rows = []
        for branches in group:
            if len(branches) != branch_count:
                raise ValueError("ragged K dimension")
            rows.append(tuple(finite_number(r, "reward") for r in branches))
        frozen.append(tuple(rows))
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class Advantages:
    per_branch: RewardCube
    prefix_mean: tuple[tuple[float, ...], ...]


def branch_advantages(rewards: Sequence[Sequence[Sequence[float]]]) -> Advantages:
    cube = validate_rewards(rewards)
    group_size = len(cube[0])
    branch_count = len(cube[0][0])
    result = []
    prefix = []
    for group in cube:
        group_advantages = []
        for g, row in enumerate(group):
            # Explicit exclusion avoids cancellation in sum(group)-own reward.
            group_advantages.append(tuple(
                row[k] - math.fsum(group[j][k] for j in range(group_size) if j != g)
                / (group_size - 1)
                for k in range(branch_count)
            ))
        result.append(tuple(group_advantages))
        prefix.append(tuple(math.fsum(row) / branch_count for row in group_advantages))
    return Advantages(tuple(result), tuple(prefix))


@dataclass(frozen=True, slots=True)
class ScoreCall:
    """One physical model generation, with the sum of its sampled-token logps.

    branch=None means prefix; an integer means suffix. call_id must be globally
    unique within the training batch and must survive branch fan-out unchanged.
    inclusion_probability is a first-order call inclusion probability, NOT the
    model action probability. It defaults to 1 for an unsampled/full-call batch.
    """

    call_id: str
    history: int
    rollout: int
    branch: int | None
    logprob_sum: float
    inclusion_probability: float = 1.0

    def __post_init__(self) -> None:
        valid_text(self.call_id, "call_id")
        if not self.call_id:
            raise ValueError("call_id must be nonempty")
        strict_integer(self.history, "history")
        strict_integer(self.rollout, "rollout")
        if self.branch is not None:
            strict_integer(self.branch, "branch")
        finite_number(self.logprob_sum, "logprob_sum")
        probability = finite_number(self.inclusion_probability, "inclusion_probability")
        if not 0 < probability <= 1:
            raise ValueError("inclusion_probability must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class LossTerm:
    call_id: str
    advantage: float
    normalization: float
    inclusion_probability: float
    coefficient: float
    logprob_sum: float

    @property
    def value(self) -> float:
        return self.coefficient * self.logprob_sum


def loss_terms(
    rewards: Sequence[Sequence[Sequence[float]]], calls: Sequence[ScoreCall],
    *, token_scale: float,
) -> tuple[LossTerm, ...]:
    cube = validate_rewards(rewards)
    advantages = branch_advantages(cube)
    scale = finite_number(token_scale, "token_scale")
    if scale <= 0:
        raise ValueError("token_scale must be > 0")
    histories, group_size, branch_count = len(cube), len(cube[0]), len(cube[0][0])
    seen = set()
    output = []
    for call in calls:
        if type(call) is not ScoreCall:
            raise ValueError("calls must be ScoreCall objects")
        if call.call_id in seen:
            raise ValueError("physical call repeated; shared prefixes must occur once")
        seen.add(call.call_id)
        if call.history >= histories or call.rollout >= group_size:
            raise ValueError("call has out-of-range history or rollout")
        if call.branch is None:
            advantage = advantages.prefix_mean[call.history][call.rollout]
            normalization = histories * group_size * scale
        else:
            if call.branch >= branch_count:
                raise ValueError("call has out-of-range branch")
            advantage = advantages.per_branch[call.history][call.rollout][call.branch]
            normalization = histories * group_size * branch_count * scale
        coefficient = -advantage / (normalization * call.inclusion_probability)
        output.append(LossTerm(
            call.call_id, advantage, normalization, call.inclusion_probability,
            coefficient, call.logprob_sum,
        ))
    return tuple(output)


def branch_policy_loss(
    rewards: Sequence[Sequence[Sequence[float]]], calls: Sequence[ScoreCall],
    *, token_scale: float,
) -> float:
    return math.fsum(term.value for term in loss_terms(rewards, calls, token_scale=token_scale))


def uniform_call_subsample(
    calls: Sequence[ScoreCall], sample_size: int, *, seed: int,
) -> tuple[ScoreCall, ...]:
    """Uniform without replacement; inclusion pi=m/M, order remains stable.

    This operates over all prefix and suffix calls together, and accepts only a
    full, previously unsampled call list. Sampled token lengths never enter pi.
    Production persists sampled IDs and RNG identity alongside the full ledger.
    """
    strict_integer(sample_size, "sample_size")
    strict_integer(seed, "seed")
    count = len(calls)
    if sample_size > count or (count > 0 and sample_size == 0):
        raise ValueError("require 1 <= sample_size <= call count for a nonempty batch")
    if len({call.call_id for call in calls}) != count:
        raise ValueError("duplicate physical call")
    if any(call.inclusion_probability != 1.0 for call in calls):
        raise ValueError("nested subsampling requires explicitly derived inclusion probabilities")
    if not count:
        return ()
    selected = sorted(random.Random(seed).sample(range(count), sample_size))
    probability = sample_size / count
    return tuple(replace(calls[i], inclusion_probability=probability) for i in selected)

