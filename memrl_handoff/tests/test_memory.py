from dataclasses import FrozenInstanceError
import hashlib
import random
import unittest

from memrl_contract.memory import (
    ByteCounter, Entry, MemoryLimits, MemoryStore, canonical_entry, serialize_entries,
)


def make_store(capacity=1000, *, max_entries=64, key_cap=40, value_cap=400, counter=None):
    return MemoryStore(MemoryLimits(capacity, max_entries, key_cap, value_cap), counter or ByteCounter())


class MemoryTests(unittest.TestCase):
    def test_exact_serialization_includes_unicode_and_delimiter(self):
        entry = Entry(1, "café", 'a"b\nc')
        self.assertEqual(canonical_entry(entry), '{"id":1,"key":"café","value":"a\\"b\\nc"}\n')
        store = make_store()
        store.write(entry.key, entry.value)
        snapshot = store.snapshot()
        encoded = canonical_entry(entry).encode("utf-8")
        self.assertEqual(snapshot.accounted_charge, len(encoded))
        self.assertEqual(snapshot.utf8_bytes, len(encoded))
        self.assertEqual(snapshot.sha256, hashlib.sha256(encoded).hexdigest())
        self.assertIn("TOY_ONLY", snapshot.counter_unit)

    def test_exact_capacity_boundary(self):
        needed = len(canonical_entry(Entry(1, "key", "value")).encode())
        fits = make_store(needed)
        self.assertTrue(fits.write("key", "value").ok)
        self.assertEqual(fits.snapshot().accounted_charge, needed)
        too_small = make_store(needed - 1)
        before = too_small.snapshot()
        self.assertEqual(too_small.write("key", "value").code, "MEMORY_FULL")
        self.assertEqual(too_small.snapshot(), before)
        self.assertEqual(too_small.audit_events(), ())

    def test_failed_update_is_atomic_and_preserves_prior_snapshots(self):
        store = make_store(80)
        store.write("k", "small")
        before = store.snapshot()
        audit = store.audit_events()
        result = store.update(1, value="x" * 100)
        self.assertEqual(result.code, "MEMORY_FULL")
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(store.audit_events(), audit)
        self.assertTrue(store.update(1, value="new").ok)
        self.assertEqual(before.entries[0].value, "small")
        with self.assertRaises(FrozenInstanceError):
            before.entries[0].value = "changed"

    def test_failed_write_does_not_consume_id(self):
        store = make_store(value_cap=4)
        self.assertEqual(store.write("k", "nope-long").code, "VALUE_LIMIT")
        self.assertEqual(store.write("k", "yes").affected_ids, (1,))

    def test_deleted_ids_are_not_recycled(self):
        store = make_store()
        store.write("first", "a")
        store.write("second", "b")
        self.assertTrue(store.delete((1,)).ok)
        self.assertEqual(store.write("third", "c").affected_ids, (3,))
        self.assertEqual(tuple(e.slot_id for e in store.snapshot().entries), (2, 3))

    def test_large_ids_are_charged(self):
        self.assertEqual(
            len(canonical_entry(Entry(10, "k", "v"))) - len(canonical_entry(Entry(9, "k", "v"))),
            1,
        )

    def test_caps_are_checked_before_commit(self):
        store = make_store(max_entries=1, key_cap=2, value_cap=3)
        self.assertEqual(store.write("too", "v").code, "KEY_LIMIT")
        self.assertEqual(store.write("k", "long").code, "VALUE_LIMIT")
        self.assertEqual(store.write("k", "v").code, "OK")
        before = store.snapshot()
        self.assertEqual(store.write("b", "v").code, "ENTRY_LIMIT")
        self.assertEqual(store.snapshot(), before)

    def test_zero_capacity_empty_arm(self):
        store = make_store(0)
        self.assertEqual(store.snapshot().accounted_charge, 0)
        self.assertEqual(store.snapshot().sha256, hashlib.sha256(b"").hexdigest())
        self.assertEqual(store.write("", "").code, "MEMORY_FULL")

    def test_strict_argument_types(self):
        store = make_store()
        for key, value in ((False, "v"), ("k", 123), ("\ud800", "v")):
            self.assertEqual(store.write(key, value).code, "BAD_ARGUMENT")
        store.write("k", "v")
        before = store.snapshot()
        for args in ((True,), (1, 1), (), [1], (1, "2")):
            self.assertEqual(store.delete(args).code, "BAD_ARGUMENT")
            self.assertEqual(store.snapshot(), before)
        self.assertEqual(store.update(True, value="v").code, "BAD_ARGUMENT")
        self.assertEqual(store.update(1, value=None).code, "BAD_ARGUMENT")
        self.assertEqual(store.update(1).code, "BAD_ARGUMENT")

    def test_delete_reports_missing_and_commits_live_set_atomically(self):
        store = make_store()
        store.write("a", "a")
        store.write("b", "b")
        result = store.delete((2, 999))
        self.assertEqual((result.ok, result.code), (True, "OK_WITH_MISSING"))
        self.assertEqual(result.affected_ids, (2,))
        self.assertEqual(result.missing_ids, (999,))
        self.assertEqual(tuple(e.slot_id for e in store.snapshot().entries), (1,))
        before = store.snapshot()
        audit = store.audit_events()
        self.assertEqual(store.delete((999,)).code, "NOT_FOUND")
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(store.audit_events(), audit)

    def test_hash_chain_and_mutation_reconstruction(self):
        store = make_store()
        hashes = [store.snapshot().sha256]
        for action in (
            lambda: store.write("a", "1"), lambda: store.write("b", "2"),
            lambda: store.update(1, value="3"), lambda: store.delete((2,)),
        ):
            self.assertTrue(action().ok)
            hashes.append(store.snapshot().sha256)
        for i, event in enumerate(store.audit_events()):
            self.assertEqual(event.before_sha256, hashes[i])
            self.assertEqual(event.after_sha256, hashes[i + 1])

    def test_whole_charge_below_additive_controls_admission(self):
        class BoundaryMergingToyCounter:
            identity = "synthetic-boundary-merge-test"
            unit = "synthetic_units"

            def count(self, text):
                return len(text.encode()) - text.count("}\n{")

        first, second = Entry(1, "a", "1"), Entry(2, "b", "2")
        additive = len(canonical_entry(first).encode()) + len(canonical_entry(second).encode())
        store = make_store(additive - 1, counter=BoundaryMergingToyCounter())
        self.assertTrue(store.write("a", "1").ok)
        self.assertTrue(store.write("b", "2").ok)
        snapshot = store.snapshot()
        self.assertEqual(snapshot.accounted_charge, additive - 1)
        self.assertEqual(snapshot.accounted_charge, snapshot.concatenated_charge)
        self.assertEqual(snapshot.additive_charge, additive)
        self.assertEqual(store.audit_events()[-1].after_charge, snapshot.accounted_charge)

    def test_whole_charge_above_additive_rejects_false_fit(self):
        class BoundarySplittingToyCounter:
            identity = "synthetic-boundary-split-test"
            unit = "synthetic_units"

            def count(self, text):
                return len(text.encode()) + text.count("}\n{")

        first, second = Entry(1, "a", "1"), Entry(2, "b", "2")
        additive = len(canonical_entry(first).encode()) + len(canonical_entry(second).encode())
        store = make_store(additive, counter=BoundarySplittingToyCounter())
        self.assertTrue(store.write("a", "1").ok)
        before, audit_before = store.snapshot(), store.audit_events()
        self.assertEqual(store.write("b", "2").code, "MEMORY_FULL")
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(store.audit_events(), audit_before)
        fits = make_store(additive + 1, counter=BoundarySplittingToyCounter())
        self.assertTrue(fits.write("a", "1").ok)
        self.assertTrue(fits.write("b", "2").ok)
        snapshot = fits.snapshot()
        self.assertEqual(snapshot.accounted_charge, additive + 1)
        self.assertEqual(snapshot.additive_charge, additive)

    def test_deletion_that_increases_whole_charge_is_rejected_atomically(self):
        class NonmonotoneToyCounter:
            identity = "synthetic-nonmonotone-test"
            unit = "synthetic_units"

            def count(self, text):
                penalty = 200 if '}\n{"id":3,' in text and '"id":2,' not in text else 0
                return len(text.encode()) + penalty

        store = make_store(150, counter=NonmonotoneToyCounter())
        for key in ("a", "b", "c"):
            self.assertTrue(store.write(key, "v").ok)
        before, audit_before = store.snapshot(), store.audit_events()
        self.assertEqual(store.delete((2,)).code, "MEMORY_FULL")
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(store.audit_events(), audit_before)

    def test_clones_isolate_mutations_deletions_audit_and_next_id(self):
        parent = make_store()
        parent.write("a", "1")
        parent.write("b", "2")
        original, original_audit = parent.snapshot(), parent.audit_events()
        first, second = parent.clone(), parent.clone()
        self.assertIs(first.counter, parent.counter)
        self.assertIs(first.snapshot().entries[0], original.entries[0])
        self.assertTrue(first.update(1, value="first-branch").ok)
        self.assertTrue(first.delete((2,)).ok)
        self.assertEqual(first.write("c", "3").affected_ids, (3,))
        self.assertEqual(first.write("d", "4").affected_ids, (4,))
        self.assertTrue(second.delete((1,)).ok)
        self.assertTrue(second.update(2, value="second-branch").ok)
        self.assertEqual(second.write("c", "independent").affected_ids, (3,))
        self.assertEqual(parent.snapshot(), original)
        self.assertEqual(parent.audit_events(), original_audit)
        self.assertEqual(tuple(e.slot_id for e in first.snapshot().entries), (1, 3, 4))
        self.assertEqual(tuple(e.slot_id for e in second.snapshot().entries), (2, 3))
        self.assertEqual(first.snapshot().entries[0].value, "first-branch")
        self.assertEqual(second.snapshot().entries[0].value, "second-branch")
        self.assertEqual(parent.write("c", "parent").affected_ids, (3,))

    def test_counter_failure_is_not_a_policy_rejection(self):
        class BadCounter:
            identity = "bad-counter"
            unit = "invalid"

            def count(self, text):
                return -1

        store = make_store(counter=BadCounter())
        with self.assertRaises(RuntimeError):
            store.write("key", "value")

    def test_random_mutations_preserve_capacity_and_transaction_invariants(self):
        rng = random.Random(183)
        store = make_store(180, max_entries=4, key_cap=8, value_cap=80)
        for _ in range(300):
            before = store.snapshot()
            audit_before = store.audit_events()
            op = rng.randrange(3)
            if op == 0:
                result = store.write("k" * rng.randrange(12), "v" * rng.randrange(100))
            elif op == 1:
                result = store.update(rng.randrange(1, 50), value="w" * rng.randrange(100))
            else:
                result = store.delete((rng.randrange(1, 50),))
            after = store.snapshot()
            self.assertLessEqual(after.accounted_charge, store.limits.capacity)
            self.assertLessEqual(len(after.entries), store.limits.max_entries)
            self.assertEqual(after.sha256, hashlib.sha256(after.serialized.encode()).hexdigest())
            if not result.ok:
                self.assertEqual(after, before)
                self.assertEqual(store.audit_events(), audit_before)

    def test_serialization_is_order_independent_but_duplicate_ids_fail(self):
        first, second = Entry(1, "a", "1"), Entry(2, "b", "2")
        self.assertEqual(serialize_entries((first, second)), serialize_entries((second, first)))
        with self.assertRaises(ValueError):
            serialize_entries((first, first))


if __name__ == "__main__":
    unittest.main()
