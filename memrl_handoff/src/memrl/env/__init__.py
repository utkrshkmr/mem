"""Writer environment. Raw history is not searchable after a chunk passes."""

from __future__ import annotations

from dataclasses import dataclass

from memrl.contracts.types import ObservationChunk, WriterObservation
from memrl.env.parser import parse_call, validate_arguments
from memrl.env.prompts import WRITER_SYSTEM, render_writer_prompt
from memrl.memory.counter import TextCounter
from memrl.memory.serialize import Entry, canonical_entry
from memrl.memory.store import MemoryLimits, MemoryStore, ToolResult

TERMINALS = frozenset({
    "normal_finish", "budget_exhausted", "model_invalid_action",
    "model_generation_limit", "infrastructure_failure",
    "implementation_invariant_failure",
})


class InvariantFailure(RuntimeError):
    """Stops the run. Callers must not relabel this as model difficulty."""


@dataclass(frozen=True, slots=True)
class StepEvent:
    event_id: str
    public_tool_result: str
    previous_snapshot_sha256: str
    next_snapshot_sha256: str
    action_status: str
    call_budget_remaining: int
    terminal: str | None


@dataclass(frozen=True, slots=True)
class AuditStep:
    event_id: str
    action_status: str
    stop_reason: str
    terminal: str | None


class WriterEnv:
    def __init__(self, chunks: tuple[str, ...], limits: MemoryLimits, counter: TextCounter, *,
                 calls_per_chunk: int, budgets: dict[str, int], task_prior: str,
                 tools: str = "full", task_instruction: str | None = None) -> None:
        if type(chunks) is not tuple or not chunks:
            raise ValueError("chunks must be a nonempty tuple of strings")
        if not all(type(chunk) is str for chunk in chunks):
            raise ValueError("chunks must be strings")
        if calls_per_chunk < 1:
            raise ValueError("calls_per_chunk must be positive")
        self.chunks = chunks
        self.counter = counter
        self.calls_per_chunk = calls_per_chunk
        self.budgets = dict(budgets)
        self.task_prior = task_prior
        self.tools = tools
        if limits.index_cap is None:
            limits = MemoryLimits(
                limits.capacity, limits.max_entries, limits.key_cap, limits.value_cap,
                budgets["index"],
            )
        self.limits = limits
        self.task_instruction = task_instruction or WRITER_SYSTEM.format(
            C=limits.capacity, K=limits.max_entries, key_cap=limits.key_cap,
            value_cap=limits.value_cap, tool_schemas=_schema_text(tools),
            public_task_prior=task_prior,
        )
        self.store = MemoryStore(limits, counter, eviction="reject")
        self.cursor = 0
        self.calls_left = calls_per_chunk
        self.loaded: dict[int, Entry] = {}
        self.last_result = ""
        self.terminal: str | None = None
        self._events = 0
        self._audit: list[AuditStep] = []
        self._history_archive_disabled = True

    def observe(self) -> WriterObservation:
        self._ensure_open()
        chunk = self.chunks[self.cursor]
        index = self.store.snapshot().serialized
        loaded = "".join(canonical_entry(self.loaded[i]) for i in sorted(self.loaded))
        observation = WriterObservation(
            task_instruction=self.task_instruction,
            chunk=ObservationChunk(
                text=chunk,
                sequence_index=self.cursor,
                is_last_in_visible_phase=self.cursor == len(self.chunks) - 1,
            ),
            memory_index=index,
            loaded_text=loaded,
            last_tool_result=self.last_result,
        )
        render_writer_prompt(
            observation, system=self.task_instruction, counter=self.counter,
            budgets=self.budgets, used=self.store.charged_tokens(),
            capacity=self.limits.capacity, n_entries=len(self.store.snapshot().entries),
            max_entries=self.limits.max_entries,
        )
        return observation

    def prompt_text(self) -> str:
        observation = self.observe()
        snap = self.store.snapshot()
        return render_writer_prompt(
            observation, system=self.task_instruction, counter=self.counter,
            budgets=self.budgets, used=snap.accounted_charge, capacity=self.limits.capacity,
            n_entries=len(snap.entries), max_entries=self.limits.max_entries,
        )

    def step(self, action_text: str, *, stop_reason: str = "eos") -> StepEvent:
        self._ensure_open()
        if stop_reason not in {"eos", "stop", "length", "error"}:
            raise InvariantFailure("unknown stop reason")
        before = self.store.snapshot_sha256()
        status = "OK"
        if stop_reason == "length":
            status = "model_generation_limit"
        parsed, error = parse_call(action_text)
        if error:
            status = error if status == "OK" else status
            return self._finish_call(before, status, stop_reason, changed=False)
        assert parsed is not None
        argument_error = validate_arguments(parsed["name"], parsed["arguments"], tools=self.tools)
        if argument_error:
            return self._finish_call(before, argument_error, stop_reason, changed=False)
        return self._execute(parsed["name"], parsed["arguments"], before, stop_reason, status)

    def advance_chunk(self, *, automatic: bool) -> StepEvent:
        self._ensure_open()
        before = self.store.snapshot_sha256()
        self._clear_ephemeral()
        consumed = list(self.chunks)
        if self.cursor < len(consumed):
            consumed[self.cursor] = ""
        self.chunks = tuple(consumed)
        self.cursor += 1
        if self.cursor >= len(self.chunks):
            self.terminal = "normal_finish"
            self.cursor = len(self.chunks)
            self.calls_left = 0
        else:
            self.calls_left = self.calls_per_chunk
        status = "AUTO_ADVANCE" if automatic else "NEXT_CHUNK"
        return self._event(before, status, "stop", self.terminal, consume=False)

    def freeze(self):
        if self.terminal != "normal_finish":
            raise InvariantFailure("freeze requires a normally finished writer")
        return self.store.snapshot()

    def clone(self) -> "WriterEnv":
        cloned = WriterEnv(
            self.chunks, self.limits, self.counter, calls_per_chunk=self.calls_per_chunk,
            budgets=self.budgets, task_prior=self.task_prior, tools=self.tools,
            task_instruction=self.task_instruction,
        )
        cloned.store = self.store.clone()
        cloned.cursor = self.cursor
        cloned.calls_left = self.calls_left
        cloned.loaded = dict(self.loaded)
        cloned.last_result = self.last_result
        cloned.terminal = self.terminal
        cloned._events = self._events
        cloned._audit = list(self._audit)
        return cloned

    def search_memory(self, query: str, top_k: int) -> str:
        entries = self.store.snapshot().entries
        needles = [part.lower() for part in query.split() if part]
        scored = []
        for entry in entries:
            hay = (entry.key + " " + entry.value).lower()
            score = sum(hay.count(needle) for needle in needles)
            scored.append((score, entry.slot_id, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [entry for score, _, entry in scored[:top_k] if score > 0 or not needles]
        return "".join(canonical_entry(entry) for entry in chosen)

    def _execute(self, name: str, arguments: dict, before: str, stop_reason: str,
                 status: str) -> StepEvent:
        if name == "next_chunk":
            self._consume()
            event = self.advance_chunk(automatic=False)
            return StepEvent(
                event.event_id, event.public_tool_result, before, event.next_snapshot_sha256,
                "NEXT_CHUNK" if status == "OK" else status, event.call_budget_remaining,
                event.terminal,
            )
        if name == "memory_list":
            return self._finish_call(before, "INDEX_OK", stop_reason, changed=False)
        if name == "memory_read":
            return self._read(arguments["slot_ids"], before, stop_reason)
        if name == "memory_search":
            text = self.search_memory(arguments["query"], arguments["top_k"])
            if self.counter.count(text) > self.budgets["last_result"]:
                return self._finish_call(before, "RESULT_LIMIT", stop_reason, changed=False)
            self.last_result = text
            return self._finish_call(before, "SEARCH_OK", stop_reason, changed=False, keep_result=True)
        if name == "memory_store_span":
            chunk = self.chunks[self.cursor]
            start, end = arguments["start_char"], arguments["end_char"]
            if end > len(chunk):
                return self._finish_call(before, "BAD_ARGUMENT", stop_reason, changed=False)
            value = chunk[start:end]
            if value not in chunk:
                raise InvariantFailure("span is not an exact substring")
            result = self.store.write(arguments["key"], value)
            return self._after_mutation(before, result, stop_reason)
        if name == "memory_write":
            result = self.store.write(arguments["key"], arguments["value"])
            return self._after_mutation(before, result, stop_reason)
        if name == "memory_update":
            kwargs = {}
            if "key" in arguments:
                kwargs["key"] = arguments["key"]
            if "value" in arguments:
                kwargs["value"] = arguments["value"]
            result = self.store.update(arguments["slot_id"], **kwargs)
            if result.ok:
                self._refresh_loaded(result.affected_ids)
            return self._after_mutation(before, result, stop_reason)
        if name == "memory_delete":
            result = self.store.delete(tuple(arguments["slot_ids"]))
            if result.ok:
                for slot_id in result.affected_ids:
                    self.loaded.pop(slot_id, None)
            return self._after_mutation(before, result, stop_reason)
        raise InvariantFailure("unhandled tool")

    def _read(self, slot_ids: list[int], before: str, stop_reason: str) -> StepEvent:
        entries = {entry.slot_id: entry for entry in self.store.snapshot().entries}
        admitted: list[Entry] = []
        omitted: list[int] = []
        running = ""
        for slot_id in slot_ids:
            if slot_id not in entries:
                omitted.append(slot_id)
                continue
            piece = canonical_entry(entries[slot_id])
            if self.counter.count(running + piece) > self.budgets["loaded"]:
                omitted.append(slot_id)
                continue
            running += piece
            admitted.append(entries[slot_id])
        self.loaded = {entry.slot_id: entry for entry in admitted}
        code = "READ_OK" if not omitted else "READ_PARTIAL"
        self.last_result = code + " omitted=" + ",".join(str(item) for item in omitted)
        return self._finish_call(before, code, stop_reason, changed=False, keep_result=True)

    def _after_mutation(self, before: str, result: ToolResult, stop_reason: str) -> StepEvent:
        self.last_result = result.public_message
        return self._finish_call(before, result.code, stop_reason, changed=result.ok, keep_result=True)

    def _refresh_loaded(self, affected: tuple[int, ...]) -> None:
        live = {entry.slot_id: entry for entry in self.store.snapshot().entries}
        for slot_id in affected:
            if slot_id in self.loaded:
                if slot_id in live:
                    self.loaded[slot_id] = live[slot_id]
                else:
                    self.loaded.pop(slot_id, None)

    def _finish_call(self, before: str, status: str, stop_reason: str, *,
                     changed: bool, keep_result: bool = False) -> StepEvent:
        if not keep_result:
            self.last_result = status
        message = self.last_result
        self._consume()
        terminal = self.terminal
        if self.calls_left == 0 and self.terminal is None:
            advanced = self.advance_chunk(automatic=True)
            reported = status if status != "OK" else "AUTO_ADVANCE"
            return StepEvent(
                advanced.event_id, message, before, advanced.next_snapshot_sha256,
                reported, advanced.call_budget_remaining, advanced.terminal,
            )
        event = self._event(before, status, stop_reason, terminal, consume=False)
        return StepEvent(
            event.event_id, message, before, event.next_snapshot_sha256,
            status, event.call_budget_remaining, event.terminal,
        )

    def _consume(self) -> None:
        if self.calls_left <= 0:
            raise InvariantFailure("call consumed without budget")
        self.calls_left -= 1

    def _clear_ephemeral(self) -> None:
        self.loaded.clear()
        self.last_result = ""

    def _event(self, before: str, status: str, stop_reason: str, terminal: str | None,
               *, consume: bool) -> StepEvent:
        if consume:
            self._consume()
        self._events += 1
        event_id = f"e{self._events}"
        self._audit.append(AuditStep(event_id, status, stop_reason, terminal))
        return StepEvent(
            event_id, self.last_result, before, self.store.snapshot_sha256(),
            status, self.calls_left, terminal,
        )

    def _ensure_open(self) -> None:
        if self.terminal is not None:
            raise InvariantFailure(f"writer already terminal: {self.terminal}")

    def mark_infrastructure_failure(self) -> None:
        self.terminal = "infrastructure_failure"

    def mark_invariant_failure(self) -> None:
        self.terminal = "implementation_invariant_failure"


def _schema_text(tools: str) -> str:
    if tools == "full":
        return "tools: write update delete read search list next_chunk store_span"
    return "tools: write delete read list next_chunk"
