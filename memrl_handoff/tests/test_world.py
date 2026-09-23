from dataclasses import FrozenInstanceError
import json
import unittest

from memrl_contract.world import Event, LedgerWorld, apply_event, continuation_counterexample, replay


class LedgerTests(unittest.TestCase):
    def test_equal_present_totals_same_future_distinguishes_histories(self):
        first, second, shared_cancel = continuation_counterexample()
        self.assertEqual(first.total, 10)
        self.assertEqual(second.total, 10)
        self.assertEqual(apply_event(first, shared_cancel).world.total, 8)
        self.assertEqual(apply_event(second, shared_cancel).world.total, 3)
        # This is a task counterexample, not a proof of information-theoretic novelty.

    def test_correction_changes_amount_subtracted_by_cancel(self):
        created = replay((Event("create", "p", 2), Event("create", "q", 8)))
        corrected = apply_event(created, Event("correction", "p", 9))
        self.assertTrue(corrected.changed)
        self.assertEqual(corrected.world.total, 17)
        cancelled = apply_event(corrected.world, Event("cancel", "p"))
        self.assertEqual(cancelled.world.total, 8)
        self.assertEqual(created.total, 10)

    def test_repeated_cancel_and_inactive_correction_are_defined_noops(self):
        cancelled = replay((Event("create", "p", 5), Event("cancel", "p")))
        again = apply_event(cancelled, Event("cancel", "p"))
        self.assertFalse(again.changed)
        self.assertEqual(again.code, "ALREADY_CANCELLED")
        self.assertIs(again.world, cancelled)
        correction = apply_event(cancelled, Event("correction", "p", 9))
        self.assertEqual(correction.code, "INACTIVE_REF")
        self.assertEqual(correction.world.total, 0)

    def test_ids_never_reused(self):
        cancelled = replay((Event("create", "p", 5), Event("cancel", "p")))
        duplicate = apply_event(cancelled, Event("create", "p", 99))
        self.assertEqual(duplicate.code, "ID_ALREADY_USED")
        self.assertEqual(duplicate.world.total, 0)

    def test_unknown_references_and_equal_correction(self):
        world = LedgerWorld()
        for event in (Event("cancel", "missing"), Event("correction", "missing", 7)):
            result = apply_event(world, event)
            self.assertFalse(result.changed)
            self.assertEqual(result.code, "UNKNOWN_REF")
        world = apply_event(world, Event("create", "p", 0)).world
        self.assertEqual(apply_event(world, Event("correction", "p", 0)).code, "UNCHANGED_VALUE")

    def test_strict_and_immutable_events(self):
        for kind, ref, value in (
            ("create", "p", True), ("create", "p", -1), ("create", "", 1),
            ("cancel", "p", 0), ("correction", "p", None), ("upsert", "p", 1),
        ):
            with self.subTest(kind=kind, ref=ref, value=value), self.assertRaises(ValueError):
                Event(kind, ref, value)
        event = Event("create", "p", 2)
        with self.assertRaises(FrozenInstanceError):
            event.value = 5
        self.assertEqual(json.loads(event.render_observation()), {"event": "create", "id": "p", "value": 2})


if __name__ == "__main__":
    unittest.main()

