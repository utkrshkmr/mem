"""Trusted CPU controller for the scripted integration fixture.

Run id is fixed to a non-paper label. Rewards come from the scripted reader.
"""

from __future__ import annotations

import json

from memrl.data.private.generate import BuiltWorld
from memrl.data.public.chunker import chunk_document
from memrl.data.public.render import render_history
from memrl.env import WriterEnv
from memrl.eval.score import score_text
from memrl.eval.scripted_reader import answer_json
from memrl.contracts.types import AnswerReceipt
from memrl.io import sha256_text

FIXTURE_ADAPTER = sha256_text("cpu-scripted-fixture-no-weights")
from memrl.memory.counter import TextCounter
from memrl.memory.store import MemoryLimits
from memrl.rl.estimator import PhysicalCall, fork_weighted_calls

FIXTURE_RUN = "cpu-scripted-fixture"


def _call(name: str, arguments: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, separators=(",", ":")) + "</tool_call>"


def drive_ledger(env: WriterEnv) -> None:
    pending: list[str] = []
    while env.terminal is None:
        observation = env.observe()
        if not pending:
            pending = [line.strip() for line in observation.chunk.text.splitlines() if line.strip().startswith("{")]
        if pending:
            payload = json.loads(pending.pop(0))
            kind = payload["event"]
            if kind == "create":
                action = _call("memory_write", {"key": str(payload["id"]), "value": str(payload["value"])})
            elif kind == "correction":
                action = _call("memory_search", {"query": str(payload["ref"]), "top_k": 1})
                env.step(action)
                if env.terminal is not None:
                    break
                found = env.last_result
                slot = None
                for line in found.splitlines():
                    if line.startswith("{"):
                        slot = json.loads(line)["id"]
                        break
                if slot is None:
                    action = _call("next_chunk", {})
                else:
                    action = _call("memory_update", {"slot_id": slot, "value": str(payload["new_value"])})
            else:
                action = _call("memory_search", {"query": str(payload["ref"]), "top_k": 1})
                env.step(action)
                if env.terminal is not None:
                    break
                slot = None
                for line in env.last_result.splitlines():
                    if line.startswith("{"):
                        slot = json.loads(line)["id"]
                        break
                if slot is None:
                    action = _call("next_chunk", {})
                else:
                    action = _call("memory_delete", {"slot_ids": [slot]})
        else:
            action = _call("next_chunk", {})
        env.step(action)


def _mean_reward(snapshot_text: str, questions, snapshot_sha: str) -> tuple[float, list]:
    rows = []
    scores = []
    for question in questions:
        text = answer_json(snapshot_text, question.prompt, question.answer_type)
        receipt = AnswerReceipt(
            snapshot_sha, question.question_id, text, sha256_text(snapshot_text + question.question_id),
            "ok",
        )
        row = score_text(
            text, question.answer, question.answer_type, failure_class="ok", receipt=receipt,
            world_id="fixture", ancestor_id="fixture", run_id=FIXTURE_RUN, branch_index=0,
        )
        rows.append(row)
        scores.append(1.0 if row.success else 0.0)
    return sum(scores) / len(scores), rows


def run_fork_fixture(world: BuiltWorld, counter: TextCounter, limits: MemoryLimits,
                     budgets: dict[str, int], *, group_size: int = 2) -> dict:
    prefix_chunks = tuple(chunk.text for chunk in world.public.chunks)
    stores = []
    for sample in range(group_size):
        env = WriterEnv(
            prefix_chunks, limits, counter, calls_per_chunk=64, budgets=budgets,
            task_prior="Track ledger identities and amounts.",
        )
        if sample == 0:
            drive_ledger(env)
        else:
            while env.terminal is None:
                env.step(_call("next_chunk", {}))
        if env.terminal != "normal_finish":
            raise RuntimeError(f"prefix did not finish: {env.terminal}")
        stores.append(env.store.clone())
    rewards = []
    score_rows = []
    calls = []
    for sample, store in enumerate(stores):
        branch_rewards = []
        calls.append(PhysicalCall(f"prefix-{sample}", 0, sample, None, -1.0, 1.0, "prefix"))
        for branch, future in enumerate(world.futures):
            rendered = render_history(future.events)
            suffix_chunks = chunk_document(rendered, {"reference": counter}, {"reference": budgets["chunk"]})
            env = WriterEnv(
                suffix_chunks, limits, counter, calls_per_chunk=64, budgets=budgets,
                task_prior="Track ledger identities and amounts.",
            )
            env.store = store.clone()
            if sample == 0:
                drive_ledger(env)
            else:
                while env.terminal is None:
                    env.step(_call("next_chunk", {}))
            snap = env.store.snapshot()
            reward, rows = _mean_reward(snap.serialized, world.questions[branch], snap.sha256)
            branch_rewards.append(reward)
            score_rows.extend(rows)
            calls.append(PhysicalCall(
                f"suffix-{sample}-{branch}", 0, sample, branch, -1.0, 1.0, "suffix",
            ))
        rewards.append(tuple(branch_rewards))
    weighted = fork_weighted_calls(
        (tuple(rewards),), tuple(calls), token_scale=256, adapter_sha256=FIXTURE_ADAPTER,
    )
    return {"rewards": rewards, "calls": weighted, "scores": score_rows, "run_id": FIXTURE_RUN}
