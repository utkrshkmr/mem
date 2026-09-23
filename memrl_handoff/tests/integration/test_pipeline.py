import unittest

from memrl.baselines.constructors import ModelRequired, event_ledger, rolling_summary
from memrl.data.private.generate import build_world
from memrl.memory.counter import CodepointCounter
from memrl.memory.store import MemoryLimits
from memrl.pipeline.controller import FIXTURE_RUN, run_fork_fixture


def budgets():
    return {"system": 8000, "chunk": 400, "index": 4000, "loaded": 2000,
            "last_result": 800, "generation": 400, "total": 20000}


class PipelineTests(unittest.TestCase):
    def test_scripted_fork_fixture(self):
        counter = CodepointCounter()
        world = build_world(
            split="dev", seed=11, index=0, n_records=4, target_length=20000,
            counters={"reference": counter, "policy": counter},
            caps={"reference": 80, "policy": 80}, k_futures=2, questions_per_future=4,
            boundary_fraction=0.5,
        )
        self.assertFalse(hasattr(world, "reason"))
        limits = MemoryLimits(4000, 32, 64, 128)
        result = run_fork_fixture(world, counter, limits, budgets())
        self.assertEqual(result["run_id"], FIXTURE_RUN)
        self.assertEqual(len(result["calls"]), 2 + 4)
        self.assertGreater(result["rewards"][0][0], result["rewards"][1][0])
        ids = [call.call_id for call in result["calls"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_event_ledger_reads_public_text_only(self):
        counter = CodepointCounter()
        chunks = ('{"event":"create","id":"p","value":2}\n',)
        store = event_ledger(chunks, MemoryLimits(1000, 8, 32, 32), counter)
        self.assertEqual(store.snapshot().entries[0].key, "p")
        with self.assertRaises(ModelRequired):
            rolling_summary()


if __name__ == "__main__":
    unittest.main()
