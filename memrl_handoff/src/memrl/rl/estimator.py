"""Fork and flat loss terms. Coefficients match the reference estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from memrl.contracts.types import SCHEMA_VERSION, WeightedCall
from memrl.contracts.common import finite_number, sha256_hex, strict_integer


def branch_advantages(rewards: tuple[tuple[tuple[float, ...], ...], ...]) -> tuple[
    tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[float, ...], ...]
]:
    if not rewards:
        raise ValueError("H must be >= 1")
    group_size = len(rewards[0])
    if group_size < 2:
        raise ValueError("G must be >= 2 for leave-one-out advantages")
    branch_count = len(rewards[0][0])
    if branch_count < 1:
        raise ValueError("K must be >= 1")
    per_branch = []
    prefix = []
    for group in rewards:
        if len(group) != group_size:
            raise ValueError("ragged G dimension")
        group_advantages = []
        for g, row in enumerate(group):
            if len(row) != branch_count:
                raise ValueError("ragged K dimension")
            group_advantages.append(tuple(
                finite_number(row[k], "reward")
                - math.fsum(finite_number(group[j][k], "reward") for j in range(group_size) if j != g)
                / (group_size - 1)
                for k in range(branch_count)
            ))
        per_branch.append(tuple(group_advantages))
        prefix.append(tuple(math.fsum(row) / branch_count for row in group_advantages))
    return tuple(per_branch), tuple(prefix)


@dataclass(frozen=True, slots=True)
class PhysicalCall:
    call_id: str
    history: int
    rollout: int
    branch: int | None
    logprob_sum: float
    inclusion_probability: float = 1.0
    phase: str = "prefix"

    def __post_init__(self) -> None:
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
        if self.phase not in {"prefix", "suffix"}:
            raise ValueError("invalid phase")
        if (self.phase == "prefix") != (self.branch is None):
            raise ValueError("prefix calls have a null branch")


def fork_weighted_calls(rewards, calls: tuple[PhysicalCall, ...], *, token_scale: float,
                        adapter_sha256: str) -> tuple[WeightedCall, ...]:
    sha256_hex(adapter_sha256, "adapter_sha256")
    per_branch, prefix = branch_advantages(rewards)
    scale = finite_number(token_scale, "token_scale")
    if scale <= 0:
        raise ValueError("token_scale must be > 0")
    histories, group_size, branch_count = len(rewards), len(rewards[0]), len(rewards[0][0])
    seen = set()
    output = []
    for call in calls:
        if type(call) is not PhysicalCall:
            raise ValueError("calls must be PhysicalCall objects")
        if call.call_id in seen:
            raise ValueError("physical call repeated; shared prefixes must occur once")
        seen.add(call.call_id)
        if call.history >= histories or call.rollout >= group_size:
            raise ValueError("call has out-of-range history or rollout")
        if call.phase == "prefix":
            advantage = prefix[call.history][call.rollout]
            factor = 1.0
            branch = None
        else:
            if call.branch is None or call.branch >= branch_count:
                raise ValueError("call has out-of-range branch")
            advantage = per_branch[call.history][call.rollout][call.branch]
            factor = 1.0 / branch_count
            branch = call.branch
        output.append(WeightedCall(
            SCHEMA_VERSION, call.call_id, call.phase, call.history, call.rollout, branch,
            adapter_sha256, advantage, factor, call.inclusion_probability, histories,
            group_size, branch_count, scale, histories * group_size, call.logprob_sum,
        ))
    return tuple(output)


def flat_weighted_calls(rewards_hk, calls: tuple[PhysicalCall, ...], *, token_scale: float,
                        adapter_sha256: str) -> tuple[WeightedCall, ...]:
    """F02: each (history, future) is an independent G-trajectory group with K=1."""
    if not rewards_hk:
        raise ValueError("H*K groups required")
    # rewards_hk[group][g] is a scalar reward. Represent as K=1 cube.
    cube = tuple(
        tuple((finite_number(reward, "reward"),) for reward in group)
        for group in rewards_hk
    )
    relabeled = []
    for call in calls:
        if call.phase != "prefix" or call.branch is not None:
            raise ValueError("flat trajectories are complete groups, not fork suffixes")
        relabeled.append(call)
    return fork_weighted_calls(cube, tuple(relabeled), token_scale=token_scale,
                               adapter_sha256=adapter_sha256)


def loss_value(calls: tuple[WeightedCall, ...]) -> float:
    return math.fsum(call.coefficient * call.call_logprob_sum for call in calls)


def uniform_subsample(calls: tuple[PhysicalCall, ...], sample_size: int, *,
                      seed: int) -> tuple[PhysicalCall, ...]:
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
    return tuple(
        PhysicalCall(
            calls[i].call_id, calls[i].history, calls[i].rollout, calls[i].branch,
            calls[i].logprob_sum, probability, calls[i].phase,
        )
        for i in selected
    )


def ddp_local_term(call: WeightedCall, world_size: int) -> float:
    strict_integer(world_size, "world_size", minimum=1)
    return call.coefficient * call.call_logprob_sum * world_size
