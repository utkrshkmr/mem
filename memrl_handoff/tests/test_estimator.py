from dataclasses import replace
from itertools import combinations, product
import math
import unittest

from memrl_contract.estimator import (
    ScoreCall, branch_advantages, branch_policy_loss, loss_terms,
    uniform_call_subsample, validate_rewards,
)


class BranchEstimatorTests(unittest.TestCase):
    def test_k_one_is_standard_rloo(self):
        rewards = (((1.0,), (0.2,), (0.4,)),)
        advantages = branch_advantages(rewards)
        expected = (0.7, -0.5, -0.2)
        for actual, target in zip(advantages.prefix_mean[0], expected):
            self.assertAlmostEqual(actual, target)
        calls = tuple(ScoreCall(str(g), 0, g, None, -(g + 1.0)) for g in range(3))
        actual = branch_policy_loss(rewards, calls, token_scale=7)
        expected_loss = -sum(expected[g] * calls[g].logprob_sum for g in range(3)) / 21
        self.assertAlmostEqual(actual, expected_loss)

    def test_per_branch_and_prefix_means(self):
        advantages = branch_advantages((((1.0, 0.0), (0.0, 0.8)),))
        self.assertEqual(advantages.per_branch, (((1.0, -0.8), (-1.0, 0.8)),))
        self.assertAlmostEqual(advantages.prefix_mean[0][0], 0.1)
        self.assertAlmostEqual(advantages.prefix_mean[0][1], -0.1)

    def test_branch_permutation_preserves_loss(self):
        rewards = (((1.0, 0.0, 0.4), (0.0, 0.8, 0.2)),)
        calls = [ScoreCall("prefix-0", 0, 0, None, -4), ScoreCall("prefix-1", 0, 1, None, -5)]
        calls.extend(ScoreCall(f"suffix-{g}-{k}", 0, g, k, -(g + k + 1)) for g in range(2) for k in range(3))
        permutation = (2, 0, 1)
        inverse = {old: new for new, old in enumerate(permutation)}
        permuted = tuple(tuple(tuple(row[k] for k in permutation) for row in group) for group in rewards)
        permuted_calls = tuple(replace(c, branch=inverse[c.branch]) if c.branch is not None else c for c in calls)
        self.assertAlmostEqual(
            branch_policy_loss(rewards, calls, token_scale=256),
            branch_policy_loss(permuted, permuted_calls, token_scale=256),
        )

    def test_prefix_is_not_multiplied_by_branch_count(self):
        calls = (ScoreCall("prefix", 0, 0, None, -3),)
        one = loss_terms((((1.0,), (0.0,)),), calls, token_scale=1)[0]
        many = loss_terms((((1.0,) * 7, (0.0,) * 7),), calls, token_scale=1)[0]
        self.assertEqual(one.coefficient, many.coefficient)
        self.assertEqual(many.normalization, 2)
        with self.assertRaisesRegex(ValueError, "physical call repeated"):
            loss_terms((((1.0, 1.0), (0.0, 0.0)),), calls + calls, token_scale=1)

    def test_suffix_normalization_contains_k(self):
        calls = (ScoreCall("suffix", 0, 0, 1, -3),)
        term = loss_terms((((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),), calls, token_scale=7)[0]
        self.assertEqual(term.normalization, 42)
        self.assertAlmostEqual(term.coefficient, -1 / 42)

    def test_multiple_histories_have_fixed_batch_normalization(self):
        cube = (((1.0,), (0.0,)), ((0.7,), (0.1,)))
        calls = (ScoreCall("h1", 1, 0, None, -1),)
        term = loss_terms(cube, calls, token_scale=5)[0]
        self.assertEqual(term.normalization, 20)
        self.assertAlmostEqual(term.coefficient, -0.6 / 20)

    def test_uniform_ht_is_exact_in_expectation_for_loss_and_every_coefficient(self):
        rewards = (((0.9, 0.2), (0.1, 0.7)),)
        calls = (
            ScoreCall("p0-a", 0, 0, None, -1),
            ScoreCall("p0-b", 0, 0, None, -17),
            ScoreCall("p1", 0, 1, None, -2),
            ScoreCall("s00", 0, 0, 0, -30),
            ScoreCall("s11", 0, 1, 1, -4),
        )
        full = loss_terms(rewards, calls, token_scale=11)
        size = 2
        subsets = tuple(combinations(range(len(calls)), size))
        probability = size / len(calls)
        totals = {c.call_id: 0.0 for c in calls}
        sampled_losses = []
        for indices in subsets:
            selected = tuple(replace(calls[i], inclusion_probability=probability) for i in indices)
            terms = loss_terms(rewards, selected, token_scale=11)
            sampled_losses.append(math.fsum(term.value for term in terms))
            for term in terms:
                totals[term.call_id] += term.coefficient / len(subsets)
        self.assertAlmostEqual(
            math.fsum(sampled_losses) / len(subsets), math.fsum(term.value for term in full),
        )
        for term in full:
            self.assertAlmostEqual(totals[term.call_id], term.coefficient)

    def test_sampler_is_reproducible_and_uses_call_inclusion(self):
        calls = tuple(ScoreCall(str(i), 0, 0, None, -i) for i in range(5))
        first = uniform_call_subsample(calls, 3, seed=101)
        self.assertEqual(first, uniform_call_subsample(calls, 3, seed=101))
        self.assertEqual(len(first), 3)
        self.assertTrue(all(c.inclusion_probability == 0.6 for c in first))
        self.assertEqual(tuple(int(c.call_id) for c in first), tuple(sorted(int(c.call_id) for c in first)))
        with self.assertRaises(ValueError):
            uniform_call_subsample(first, 2, seed=101)
        with self.assertRaises(ValueError):
            uniform_call_subsample(calls, 0, seed=101)
        self.assertEqual(uniform_call_subsample((), 0, seed=101), ())

    def test_invalid_shapes_and_numerics_fail(self):
        bad_cubes = ((), (((), ()),), (((1.0,),),), (((1.0,), (0.0, 0.2)),), (((float("nan"),), (0.0,)),))
        for cube in bad_cubes:
            with self.subTest(cube=cube), self.assertRaises(ValueError):
                validate_rewards(cube)
        with self.assertRaises(ValueError):
            ScoreCall("x", True, 0, None, -1)
        with self.assertRaises(ValueError):
            ScoreCall("x", 0, 0, None, -1, inclusion_probability=0)
        with self.assertRaises(ValueError):
            loss_terms((((1.0,), (0.0,)),), (ScoreCall("x", 0, 0, 1, -1),), token_scale=1)
        with self.assertRaises(ValueError):
            loss_terms((((1.0,), (0.0,)),), (), token_scale=0)


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def bernoulli_probability(action, probability):
    return probability if action else 1 - probability


def feature(prefix_action, branch):
    return (1.2 if prefix_action else -0.7) + 0.4 * branch


def reward(prefix_action, suffix_action, branch):
    # Deliberately includes an interaction and a branch-dependent trade-off.
    if branch == 0:
        return 1.0 if prefix_action == suffix_action else 0.1
    return 0.8 if (prefix_action == 0 and suffix_action == 1) else 0.2 * prefix_action


def exact_expected_objective(theta, branches):
    """One rollout's true objective; exact sum, no sampled estimator involved."""
    prefix_probability = sigmoid(theta)
    expectation = 0.0
    for prefix_action in (0, 1):
        for branch in range(branches):
            conditional_probability = sigmoid(theta * feature(prefix_action, branch))
            for suffix_action in (0, 1):
                probability = (
                    bernoulli_probability(prefix_action, prefix_probability)
                    * bernoulli_probability(suffix_action, conditional_probability)
                )
                expectation += probability * reward(prefix_action, suffix_action, branch) / branches
    return expectation


class ExhaustiveGradientTests(unittest.TestCase):
    def expected_loss_gradient(self, theta, branches, token_scale):
        """Enumerate ALL G=2 prefix/suffix sample combinations (16 or 64)."""
        group_size = 2
        prefix_probability = sigmoid(theta)
        expected_gradient = 0.0
        probability_mass = 0.0
        for prefix_actions in product((0, 1), repeat=group_size):
            for suffix_flat in product((0, 1), repeat=group_size * branches):
                suffix_actions = tuple(
                    suffix_flat[g * branches:(g + 1) * branches] for g in range(group_size)
                )
                probability = 1.0
                calls, scores, rewards = [], {}, []
                for g in range(group_size):
                    a = prefix_actions[g]
                    pa = bernoulli_probability(a, prefix_probability)
                    probability *= pa
                    prefix_id = f"prefix-{g}"
                    calls.append(ScoreCall(prefix_id, 0, g, None, math.log(pa)))
                    scores[prefix_id] = a - prefix_probability
                    row = []
                    for k in range(branches):
                        b = suffix_actions[g][k]
                        x = feature(a, k)
                        conditional_probability = sigmoid(theta * x)
                        pb = bernoulli_probability(b, conditional_probability)
                        probability *= pb
                        suffix_id = f"suffix-{g}-{k}"
                        calls.append(ScoreCall(suffix_id, 0, g, k, math.log(pb)))
                        scores[suffix_id] = x * (b - conditional_probability)
                        row.append(reward(a, b, k))
                    rewards.append(tuple(row))
                terms = loss_terms((tuple(rewards),), tuple(calls), token_scale=token_scale)
                conditional_gradient = math.fsum(term.coefficient * scores[term.call_id] for term in terms)
                expected_gradient += probability * conditional_gradient
                probability_mass += probability
        self.assertAlmostEqual(probability_mass, 1.0, places=12)
        return expected_gradient

    def test_estimator_gradient_equals_exact_expected_objective_derivative(self):
        # This checks the actual expectation identity, not a duplicate formula.
        # Rewards are held fixed while differentiating each sampled log-probability.
        epsilon = 1e-5
        for branches, theta, token_scale in product((1, 2), (-0.9, 0.2, 1.3), (1.0, 7.0)):
            with self.subTest(K=branches, theta=theta, Z=token_scale):
                exact_derivative = (
                    exact_expected_objective(theta + epsilon, branches)
                    - exact_expected_objective(theta - epsilon, branches)
                ) / (2 * epsilon)
                observed = self.expected_loss_gradient(theta, branches, token_scale)
                self.assertAlmostEqual(observed, -exact_derivative / token_scale, places=9)


if __name__ == "__main__":
    unittest.main()

