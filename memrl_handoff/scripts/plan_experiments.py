"""Expand a selected research suite into a draft plan; never execute jobs."""
import argparse
import copy
import math
import sys

from config_contract import ROOT, atomic_json, fingerprint, load_json, resolve, validate


def build_plan(suite_name, costs=None, cap=1500.0):
    registry = load_json(ROOT / "configs/experiment_registry.json")
    if suite_name not in registry["suites"]:
        raise ValueError("unknown suite")
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("budget cap must be positive finite")
    if cap > registry["project_gpu_hour_cap"]:
        raise ValueError("requested budget exceeds declared project envelope")
    suite = registry["suites"][suite_name]
    measured = costs is not None and costs.get("status") == "measured"
    if costs is not None and not measured:
        raise ValueError("cost template is unmeasured; run the GPU profiling pilot first")
    factor = 1.25 if costs is None else costs.get("safety_factor", 1.25)
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor < 1:
        raise ValueError("invalid cost safety factor")
    jobs = []
    for arm in suite["arms"]:
        original = resolve(ROOT / f"configs/profiles/{arm}.json")
        for capacity in suite["capacities"]:
            for seed in suite["seeds"]:
                c = copy.deepcopy(original)
                c["run"].update(study=suite["study"], seed=seed, development_only=suite["development_only"])
                c["memory"]["capacity_ref_tokens"] = capacity
                c["rl"]["iterations"] = suite["iterations"]
                validate(c)
                key = f"{arm}:C{capacity}:I{suite['iterations']}"
                estimate = None if costs is None else costs.get("costs", {}).get(key)
                if estimate is not None and (type(estimate) not in (int, float)
                                            or not math.isfinite(estimate) or estimate <= 0):
                    raise ValueError(f"invalid measured cost for {key}")
                jobs.append({"job_id": f"{suite['study']}_{arm}_C{capacity}_s{seed}",
                             "arm": arm, "seed": seed, "capacity": capacity,
                             "iterations": suite["iterations"] if c["run"]["mode"] == "train" else 0,
                             "mode": c["run"]["mode"], "config_sha256": fingerprint(c),
                             "required_gates": suite["required_gates"],
                             "projected_allocated_gpu_hours_with_reserve": None if estimate is None else estimate * factor,
                             "config": c})
    known = all(j["projected_allocated_gpu_hours_with_reserve"] is not None for j in jobs)
    total = sum(j["projected_allocated_gpu_hours_with_reserve"] for j in jobs) if known else None
    return {"status": "draft_plan_only", "launches_jobs": False, "suite": suite_name,
            "job_count": len(jobs), "budget_cap": cap, "costs_complete": known,
            "projected_allocated_gpu_hours_with_reserve": total,
            "fits_budget": None if total is None else total <= cap,
            "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("pilot", "core", "ablation"), required=True)
    parser.add_argument("--costs")
    parser.add_argument("--budget-gpu-hours", type=float, default=1500)
    parser.add_argument("--require-budgeted", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        plan = build_plan(args.suite, load_json(args.costs) if args.costs else None, args.budget_gpu_hours)
        if args.require_budgeted and (not plan["costs_complete"] or not plan["fits_budget"]):
            raise ValueError("missing measured costs or plan exceeds budget; no launch is authorized")
        atomic_json(args.output, plan)
        print(f"Draft {args.suite} plan: {plan['job_count']} jobs; measured costs complete={plan['costs_complete']}; no jobs launched.")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
