import math
import unittest

from memrl_contract.estimator import ScoreCall, branch_policy_loss
from memrl.rl.estimator import PhysicalCall, fork_weighted_calls, loss_value, uniform_subsample
from memrl.rl.masks import completion_mask, masked_logprob_sum
from memrl.rl.selection import select_checkpoint

ADAPTER = "ab" * 32


class LossTests(unittest.TestCase):
    def test_matches_reference_and_rejects_duplicate_prefix(self):
        rewards = (((1.0, 0.0), (0.0, 0.8)),)
        calls = (
            PhysicalCall("p0", 0, 0, None, -4.0, phase="prefix"),
            PhysicalCall("p1", 0, 1, None, -5.0, phase="prefix"),
            PhysicalCall("s00", 0, 0, 0, -2.0, phase="suffix"),
            PhysicalCall("s11", 0, 1, 1, -3.0, phase="suffix"),
        )
        weighted = fork_weighted_calls(rewards, calls, token_scale=256, adapter_sha256=ADAPTER)
        reference_calls = (
            ScoreCall("p0", 0, 0, None, -4.0),
            ScoreCall("p1", 0, 1, None, -5.0),
            ScoreCall("s00", 0, 0, 0, -2.0),
            ScoreCall("s11", 0, 1, 1, -3.0),
        )
        self.assertAlmostEqual(loss_value(weighted), branch_policy_loss(rewards, reference_calls, token_scale=256))
        self.assertEqual(weighted[0].branch_factor, 1)
        self.assertAlmostEqual(weighted[2].branch_factor, 0.5)
        with self.assertRaisesRegex(ValueError, "physical call repeated"):
            fork_weighted_calls(rewards, calls[:1] + calls[:1], token_scale=256, adapter_sha256=ADAPTER)

    def test_g1_and_k0_rejected(self):
        with self.assertRaisesRegex(ValueError, "G must be"):
            fork_weighted_calls((((1.0,),),), (), token_scale=1, adapter_sha256=ADAPTER)
        with self.assertRaisesRegex(ValueError, "K must be"):
            fork_weighted_calls((((), ()),), (), token_scale=1, adapter_sha256=ADAPTER)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            fork_weighted_calls(
                (((1.0,), (0.0,)),), (), token_scale=1, adapter_sha256="0" * 64,
            )

    def test_subsample_inclusion(self):
        calls = tuple(PhysicalCall(str(i), 0, 0, None, -float(i), phase="prefix") for i in range(4))
        # G dimension is not checked by the sampler.
        selected = uniform_subsample(calls, 2, seed=3)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(math.isclose(call.inclusion_probability, 0.5) for call in selected))

    def test_mask_excludes_prompt(self):
        mask = completion_mask(3, 2, 5)
        self.assertEqual(mask, [False, False, False, True, True])
        self.assertAlmostEqual(masked_logprob_sum([1, 1, 1, 0.5, -0.25], mask), 0.25)

    def test_selector_blocks_final_benchmarks(self):
        with self.assertRaisesRegex(ValueError, "final-benchmark"):
            select_checkpoint([{"dev_score": 1, "iteration": 1, "source": "longmemeval"}], split="development")
        chosen = select_checkpoint(
            [{"dev_score": 0.2, "iteration": 3, "source": "dev"},
             {"dev_score": 0.2, "iteration": 1, "source": "dev"}],
            split="development",
        )
        self.assertEqual(chosen["iteration"], 1)


if __name__ == "__main__":
    unittest.main()
