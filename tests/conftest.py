"""Shared test helpers: `record` writes measured values to reports/tests/<id>.json, and every
test outcome is appended to reports/tests/results.jsonl with its plan ID (PLAN.md §9)."""
import datetime
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "tests"

# Appendix A tests keep their names; this maps them to plan IDs.
APPENDIX_IDS = {
    "test_answers_match_text_reparse": "T1.3", "test_determinism_and_independence": "T1.3",
    "test_prefix_does_not_depend_on_future_generation": "T1.3", "test_state_roundtrip": "T1.3",
    "test_answers_not_guessable": "T1.3", "test_futures_touch_prefix_transactions": "T1.3",
    "test_estimator_unbiased": "T1.6", "test_weights_sum_rules": "T1.6",
    "test_fixed_baseline_variance_formula": "T1.6",
    "test_variance_accumulator_recovers_components": "T1.7", "test_torch_accumulator_matches_numpy": "T1.7",
    "test_linear_surface_choices_agree": "T1.8", "test_saturating_surface_moves_choice": "T1.8",
    "test_profile_once_model_a": "T1.8", "test_curves": "T1.9",
    "test_logprob_sums_match_naive": "T1.10", "test_microbatch_partition_invariance": "T1.10",
    "test_gradient_matches_naive_objective": "T1.10", "test_only_lora_trainable_and_dropout_zero": "T1.10",
    "test_packing_respects_budget": "T1.10", "test_sign_of_update": "T1.10",
}


def plan_id(test_name: str) -> str | None:
    base = test_name.split("[", 1)[0]
    m = re.match(r"test_(T\d+)_(\d+)", base)
    return f"{m[1]}.{m[2]}" if m else APPENDIX_IDS.get(base)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@pytest.fixture
def record(request):
    def _record(test_id: str, **values):
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / f"{test_id}.json"
        path.write_text(json.dumps({"id": test_id, "nodeid": request.node.nodeid, "at": _now(), **values},
                                   indent=2, sort_keys=True, default=str))
    return _record


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        REPORTS.mkdir(parents=True, exist_ok=True)
        name = report.nodeid.split("::")[-1]
        with open(REPORTS / "results.jsonl", "a") as f:
            f.write(json.dumps({"at": _now(), "id": plan_id(name), "nodeid": report.nodeid,
                                "when": report.when, "outcome": report.outcome,
                                "duration_s": round(report.duration, 3)}) + "\n")
