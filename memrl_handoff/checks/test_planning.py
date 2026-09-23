"""Negative-contract and planning checks; no hardware or model dependencies."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config_contract import load_json, merge_strict, production_blockers, resolve, validate
from plan_experiments import build_plan


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = load_json(ROOT / "configs/base.json")

    def test_all_declared_profiles_resolve(self):
        for path in sorted((ROOT / "configs/profiles").glob("*.json")):
            with self.subTest(path=path.name):
                self.assertEqual(resolve(path)["run"]["arm"], path.stem)

    def test_unknown_key_and_wrong_type_fail(self):
        for overlay in ({"reader": {"gold": True}}, {"reader": {"trainable": "false"}},
                        {"rl": {"trajectories_per_group": True}}):
            with self.assertRaises(ValueError):
                merge_strict(self.base, overlay)

    def test_scientific_violations_fail(self):
        overlays = [
            {"reader": {"trainable": True}},
            {"continuation": {"reveal_future_before_fork": True}},
            {"continuation": {"futures_per_prefix": 0}},
            {"continuation": {"reuse_prefix": False}},
            {"memory": {"capacity_ref_tokens": 8192}},
            {"rl": {"trajectories_per_group": 1}},
            {"rl": {"normalize_advantages": True}},
            {"writer_budget": {"generation": 8192}},
            {"hardware": {"rollout_physical_gpus": [1, 3]}},
            {"data": {"allow_external_test_for_selection": True}},
        ]
        for overlay in overlays:
            with self.subTest(overlay=overlay), self.assertRaises(ValueError):
                validate(merge_strict(self.base, overlay))

    def test_templates_are_not_production_ready(self):
        problems = production_blockers(self.base)
        self.assertTrue(any("writer_revision" in s for s in problems))
        self.assertTrue(any("runtime_lock" in s for s in problems))

    def test_production_cli_fails_without_writing_success(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "must-not-exist.json"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/resolve_config.py"),
                                     "--profile", str(ROOT / "configs/profiles/F03_fork.json"),
                                     "--production", "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("blocked", result.stderr)
            self.assertFalse(out.exists())

    def test_duplicate_keys_and_nonfinite_json_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_json(path)


class PlannerTests(unittest.TestCase):
    def test_pilot_does_not_claim_known_costs(self):
        p = build_plan("pilot")
        self.assertEqual(p["job_count"], 4)
        self.assertFalse(p["launches_jobs"])
        self.assertIsNone(p["fits_budget"])
        self.assertIsNone(p["projected_allocated_gpu_hours_with_reserve"])

    def test_core_has_three_arms_and_three_seeds(self):
        p = build_plan("core")
        self.assertEqual(p["job_count"], 9)
        self.assertEqual(len({j["job_id"] for j in p["jobs"]}), 9)
        self.assertEqual({j["capacity"] for j in p["jobs"]}, {1024})

    def test_costs_include_seed_multiplicity_and_reserve(self):
        # Deliberately synthetic UNIT-TEST costs, not a hardware estimate.
        costs = {"status": "measured", "safety_factor": 1.25,
                 "costs": {f"{arm}:C1024:I200": 10 for arm in
                           ("F01_singlefuture", "F02_flat", "F03_fork")}}
        p = build_plan("core", costs, 100)
        self.assertEqual(p["projected_allocated_gpu_hours_with_reserve"], 112.5)
        self.assertFalse(p["fits_budget"])

    def test_missing_cost_not_zero(self):
        p = build_plan("core", {"status": "measured", "costs": {}})
        self.assertFalse(p["costs_complete"])
        self.assertIsNone(p["projected_allocated_gpu_hours_with_reserve"])

    def test_unmeasured_and_nonfinite_costs_fail(self):
        with self.assertRaises(ValueError):
            build_plan("pilot", {"status": "unmeasured_template", "costs": {}})
        with self.assertRaises(ValueError):
            build_plan("pilot", {"status": "measured", "costs": {"F01_singlefuture:C1024:I20": float("nan")}})
        with self.assertRaises(ValueError):
            build_plan("pilot", cap=1501)


if __name__ == "__main__":
    unittest.main()
