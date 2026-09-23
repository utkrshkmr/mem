import unittest

from memrl.contracts.types import (
    Generation, ModelLock, ObservationChunk, TokenizedCall, WeightedCall, WriterObservation,
)
from memrl.memory.counter import CodepointCounter, NonMonotoneCounter
from memrl.memory.store import MemoryLimits, MemoryStore


class ContractTests(unittest.TestCase):
    def test_reader_rejects_writer_adapter(self):
        with self.assertRaisesRegex(ValueError, "writer adapter on reader"):
            TokenizedCall((1, 2), 4, "a" * 64, 0, "reader", "r1")

    def test_logprob_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "corrupt behavior-logprob length"):
            Generation((1,), (), "x", "eos", None, "r", "writer")

    def test_prefix_factor(self):
        with self.assertRaisesRegex(ValueError, "branch_factor"):
            WeightedCall(
                3, "c", "prefix", 0, 0, None, "b" * 64, 0.0, 0.5, 1.0, 1, 2, 1, 256, 2, -1.0,
            )

    def test_unpinned_model_lock(self):
        with self.assertRaises(ValueError):
            ModelLock("Qwen/Qwen2.5-7B-Instruct", "main", "Qwen/Qwen2.5-7B-Instruct", "main", "c" * 64)

    def test_observation_roundtrip_excludes_gold(self):
        obs = WriterObservation("instr", ObservationChunk("hello", 0, False), "", "", "")
        self.assertNotIn("gold", obs.payload())


class StoreTests(unittest.TestCase):
    def _store(self, capacity=80, **kwargs):
        limits = MemoryLimits(capacity, 8, 20, 40, index_cap=kwargs.pop("index_cap", None))
        return MemoryStore(limits, kwargs.pop("counter", CodepointCounter()), **kwargs)

    def test_failed_write_is_atomic(self):
        store = self._store()
        self.assertTrue(store.write("a", "1").ok)
        before = store.public_snapshot()
        self.assertFalse(store.write("a" * 30, "1").ok)
        self.assertEqual(store.public_snapshot(), before)
        self.assertEqual(store.write("b", "2").affected_ids[0], 2)

    def test_one_token_over_capacity(self):
        store = self._store(capacity=40)
        self.assertTrue(store.write("k", "v").ok)
        blob = store.snapshot().serialized
        # Fill until one more code point would overflow, then reject.
        while store.charged_tokens() + 15 < 40:
            store.write("k", "v")
        before = store.public_snapshot()
        self.assertFalse(store.write("long-key", "long-value-overflow").ok)
        self.assertEqual(store.public_snapshot(), before)
        self.assertLessEqual(store.charged_tokens(), 40)

    def test_delete_can_increase_charge(self):
        counter = NonMonotoneCounter()
        store = MemoryStore(MemoryLimits(10, 4, 20, 30), counter)
        self.assertTrue(store.write("a", "DROPME").ok)
        self.assertTrue(store.write("b", "0123456789ABCDEF").ok)
        self.assertEqual(store.charged_tokens(), 1)
        self.assertFalse(store.delete((1,)).ok)
        self.assertEqual(store.snapshot().entries[0].value, "DROPME")

    def test_clone_isolation(self):
        store = self._store()
        store.write("k", "v")
        cloned = store.clone()
        cloned.write("q", "z")
        self.assertEqual(len(store.snapshot().entries), 1)
        self.assertEqual(len(cloned.snapshot().entries), 2)

    def test_bool_slot_rejected(self):
        store = self._store()
        store.write("k", "v")
        self.assertFalse(store.update(True, value="x").ok)  # type: ignore[arg-type]

    def test_random_sequences_hold_invariants(self):
        import random
        rng = random.Random(0)
        store = self._store(capacity=200)
        for _ in range(500):
            roll = rng.randrange(3)
            before = store.public_snapshot()
            if roll == 0:
                result = store.write(rng.choice("abcd"), str(rng.randrange(10)))
            elif roll == 1 and store.snapshot().entries:
                slot = store.snapshot().entries[0].slot_id
                result = store.update(slot, value=str(rng.randrange(10)))
            elif store.snapshot().entries:
                slot = store.snapshot().entries[-1].slot_id
                result = store.delete((slot,))
            else:
                result = store.delete((1,))
            if not result.ok:
                self.assertEqual(store.public_snapshot(), before)
            self.assertLessEqual(store.charged_tokens(), 200)
            self.assertEqual(len({e.slot_id for e in store.snapshot().entries}), len(store.snapshot().entries))


if __name__ == "__main__":
    unittest.main()
