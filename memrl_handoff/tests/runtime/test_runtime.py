import json
import unittest

from memrl.analysis.aggregate import (
    bootstrap_mean_interval, paired_difference, table_cell, validate_completeness,
)
from memrl.env import WriterEnv
from memrl.env.parser import parse_call
from memrl.eval.cache import reader_cache_key
from memrl.eval.score import score_typed
from memrl.memory.counter import CodepointCounter
from memrl.memory.store import MemoryLimits
from memrl.runtime.lease import LeaseError, LeaseRegistry
from memrl.runtime.publish import publish_adapter, publish_json
from memrl.runtime.resume import recover_iteration, write_checkpoint


def budgets():
    return {"system": 8000, "chunk": 2000, "index": 2000, "loaded": 2000,
            "last_result": 800, "generation": 400, "total": 20000}


class EvalAndRuntimeTests(unittest.TestCase):
    def test_score_rejects_substring_and_bool(self):
        self.assertFalse(score_typed("the answer is 7", 7, "integer")[0])
        self.assertFalse(score_typed(True, 1, "integer")[0])
        self.assertTrue(score_typed(None, None, "unknown")[0])

    def test_reader_cache_changes_with_snapshot(self):
        left = reader_cache_key(
            prompt_ids=(1, 2), model_revision="rev", adapter_sha256=None,
            snapshot_sha256="a" * 64, template_sha256="b" * 64,
        )
        right = reader_cache_key(
            prompt_ids=(1, 2), model_revision="rev", adapter_sha256=None,
            snapshot_sha256="c" * 64, template_sha256="b" * 64,
        )
        self.assertNotEqual(left, right)
        with self.assertRaisesRegex(ValueError, "writer adapter on reader"):
            reader_cache_key(
                prompt_ids=(1,), model_revision="rev", adapter_sha256="d" * 64,
                snapshot_sha256="a" * 64, template_sha256="b" * 64,
            )

    def test_parser_rejects_extra_fields(self):
        text = '<tool_call>{"name":"memory_write","arguments":{"key":"a","value":"b","gold":"1"}}</tool_call>'
        parsed, error = parse_call(text)
        self.assertIsNone(error)
        from memrl.env.parser import validate_arguments
        self.assertEqual(validate_arguments(parsed["name"], parsed["arguments"], tools="full"), "BAD_ARGUMENT")

    def test_prompt_stable_when_private_gold_changes(self):
        counter = CodepointCounter()
        limits = MemoryLimits(500, 8, 40, 80)
        env = WriterEnv(("alpha secret-history",), limits, counter, calls_per_chunk=2,
                        budgets=budgets(), task_prior="Keep facts.")
        first = env.prompt_text()
        private = {"answer": "CHANGED", "split": "test", "gold": "nope"}
        self.assertNotIn(private["answer"], first)
        self.assertEqual(env.prompt_text(), first)
        self.assertNotIn("CHANGED", first)

    def test_past_chunk_is_not_searchable(self):
        counter = CodepointCounter()
        limits = MemoryLimits(800, 8, 40, 80)
        env = WriterEnv(("UNIQUEPAST", "second"), limits, counter, calls_per_chunk=2,
                        budgets=budgets(), task_prior="Keep facts.")
        env.step('<tool_call>{"name":"next_chunk","arguments":{}}</tool_call>')
        found = env.search_memory("UNIQUEPAST", 3)
        self.assertNotIn("UNIQUEPAST", found)
        self.assertEqual(env.chunks[0], "")

    def test_bootstrap_and_completeness(self):
        manifest = [{"world_id": "w", "question_id": "q", "branch_index": 0}]
        rows = [{"world_id": "w", "question_id": "q", "branch_index": 0, "failure_class": "ok",
                 "success": True, "infrastructure": False, "ancestor_id": "w", "arm": "A"}]
        validate_completeness(rows, manifest)
        pairs = [("w", 0.0)]
        interval = bootstrap_mean_interval(pairs, replicates=200, seed=1)
        self.assertEqual(interval["estimate"], 0.0)
        self.assertEqual(table_cell(
            estimate=None, low=None, high=None, n_worlds=0, n_seeds=0, metric="acc",
            contrast_id="F03-F02", source_sha256="0" * 64, status="not_run",
        )["status"], "not_run")
        with self.assertRaises(ValueError):
            paired_difference({"a": 1.0}, {"b": 1.0})

    def test_lease_publish_and_resume(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leases = LeaseRegistry(root / "leases")
            leases.acquire("GPU-1", "owner-a", 30)
            with self.assertRaises(LeaseError):
                leases.acquire("GPU-1", "owner-b", 30)
            leases.release("GPU-1", "owner-a")
            publish_json(root / "jobs", "job.json", {"schema_version": 3})
            with self.assertRaises(FileExistsError):
                publish_json(root / "jobs", "job.json", {"schema_version": 3})
            manifest = publish_adapter(root / "adapters", {"w.bin": b"abc"}, version=1, base_revision="r" * 40)
            self.assertEqual(len(manifest["adapter_sha256"]), 64)
            write_checkpoint(root / "ckpt", 4, {"rng": 1})
            self.assertEqual(recover_iteration(root / "ckpt"), 5)
            partial = root / "jobs" / ".result.partial"
            partial.write_text("", encoding="utf-8")
            from memrl.runtime.publish import load_complete
            with self.assertRaises(ValueError):
                load_complete(partial)


if __name__ == "__main__":
    unittest.main()
