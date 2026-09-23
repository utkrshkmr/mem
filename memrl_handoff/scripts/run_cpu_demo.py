"""Small reference demonstration, not a model or a paper experiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from memrl_contract.world import apply_event, continuation_counterexample
from memrl_contract.estimator import branch_advantages


def main():
    first, second, future = continuation_counterexample()
    after = [apply_event(w, future).world.total for w in (first, second)]
    # Test-only fixed numbers; never write them to a research result directory.
    advantages = branch_advantages([[[1.0, 0.0], [0.0, 1.0]]])
    result = {"status": "CPU_fixture_only", "models_used": False,
              "same_present_totals": [first.total, second.total],
              "shared_future_event": future.render_observation(),
              "different_future_totals": after,
              "fixture_per_branch_advantages": advantages.per_branch,
              "fixture_prefix_mean_advantages": advantages.prefix_mean}
    assert result["same_present_totals"] == [10, 10] and after == [8, 3]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
