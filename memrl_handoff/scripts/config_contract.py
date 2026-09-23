"""Strict CPU planning helpers. This is not the production launch validator."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARM_MODES = {
    "F00_static": "static", "F01_singlefuture": "single",
    "F02_flat": "flat", "F03_fork": "fork",
    "F04_event_ledger": "event_ledger", "F05_duplicate_future": "duplicate_future",
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_pairs, parse_constant=bad_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_strict(base, overlay, prefix=""):
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise ValueError("configuration roots must be objects")
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        dotted = f"{prefix}.{key}".lstrip(".")
        if key not in base:
            raise ValueError(f"unknown configuration key: {dotted}")
        old = base[key]
        if isinstance(old, dict):
            if not isinstance(value, dict):
                raise ValueError(f"expected object: {dotted}")
            result[key] = merge_strict(old, value, dotted)
        else:
            # Null template paths and native_window are resolved later.
            polymorphic = dotted == "model.native_window"
            number = type(old) in (int, float) and type(value) in (int, float)
            if old is not None and not polymorphic and not number and type(old) is not type(value):
                raise ValueError(f"wrong type: {dotted}")
            result[key] = copy.deepcopy(value)
    return result


def validate(config):
    c = config
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def positive_int(value):
        return type(value) is int and value > 0

    require(c["spec_version"] == 3, "spec_version must be 3")
    require(c["run"]["protocol"] == "continuation_blind_v3", "unexpected protocol")
    require(c["run"]["arm"] in ARM_MODES, "unknown arm")
    require(type(c["run"]["seed"]) is int and c["run"]["seed"] >= 0, "invalid seed")
    require(c["run"]["mode"] in ("train", "evaluate_baseline"), "invalid run mode")
    for name in ("writer_budget", "reader_budget"):
        budget = c[name]
        require(all(positive_int(v) for v in budget.values()), f"invalid {name} values")
        require(sum(v for k, v in budget.items() if k != "total") <= budget["total"],
                f"{name} sum exceeds total")
    cap = c["memory"]["capacity_ref_tokens"]
    require(type(cap) is int and 0 < cap <= 4096, "primary capacity outside (0,4096]")
    require(c["memory"]["include_visible_metadata_in_capacity"] is True, "metadata must be charged")
    require(c["memory"]["expose_raw_history_search"] is False, "raw archive forbidden")
    require(c["environment"]["writer_question_visibility"] == "blind", "writer must be blind")
    require(c["reader"]["trainable"] is False, "reader must be frozen")
    require(c["reader"]["memory_view"] == "all_retained", "primary reader must see retained store")
    require(c["retrieval"]["enabled"] is False, "retrieval is secondary only")
    require(c["data"]["allow_external_test_for_selection"] is False, "test selection forbidden")
    rl = c["rl"]
    require(type(rl["trajectories_per_group"]) is int and rl["trajectories_per_group"] >= 2,
            "RLOO needs G>=2")
    for key in ("groups_per_iteration", "iterations", "selected_calls_per_iteration", "estimator_scale"):
        require(positive_int(rl[key]), f"invalid rl.{key}")
    require(rl["estimator"] == "rloo_pg", "reference estimator must be rloo_pg")
    require(rl["normalize_advantages"] is False and rl["kl_beta"] == 0, "changed reference objective")
    require(rl["writer_temperature"] == 1 and rl["top_p"] == 1 and rl["top_k"] == -1
            and rl["repetition_penalty"] == 1, "unsupported sampling distribution")
    require(rl["lora"]["dropout"] == 0, "LoRA dropout must be zero")
    require(rl["reward"] == {"task_weight": 1.0, "call_penalty": 0.0,
                             "format_penalty": 0.0, "gold_retention_shaping": 0.0},
            "primary reward must be pure task score")
    b = c["continuation"]
    require(positive_int(b["futures_per_prefix"]), "K must be positive integer")
    require(b["mode"] == ARM_MODES.get(c["run"]["arm"]), "arm/mode mismatch")
    require(b["future_rng_independent_of_writer_actions"] is True, "future kernel must be exogenous")
    require(b["share_futures_across_g"] is True and b["share_queries_across_g"] is True,
            "matched branches require common hidden material")
    require(b["reveal_future_before_fork"] is False and b["raw_history_archive_access"] is False,
            "future or archive leakage")
    require(b["allocator"]["enabled"] is False, "optional allocator needs a new validated protocol")
    if b["mode"] == "static":
        require(b["enabled"] is False and b["futures_per_prefix"] == 1 and not b["reuse_prefix"]
                and b["reward_endpoint"] == "current_mean", "invalid static arm")
    else:
        require(b["enabled"] is True and b["reward_endpoint"] == "future_mean", "invalid future arm")
        require(b["reuse_prefix"] == (b["mode"] in ("fork", "duplicate_future", "event_ledger")),
                "wrong prefix reuse setting")
    if b["mode"] == "single":
        require(b["futures_per_prefix"] == 1, "single-future arm requires K=1")
    require((c["run"]["mode"] == "evaluate_baseline") == (b["mode"] == "event_ledger"),
            "event ledger is the nontraining arm")
    mix = list(b["target_age_mixture"].values())
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in mix)
            and math.isclose(sum(mix), 1.0), "future mixture must be probabilities summing to one")
    require(all(type(v) in (int, float) and 0 < v < 1 for v in b["prefix_boundary_event_fractions"])
            and len(b["prefix_boundary_event_fractions"]) > 0, "invalid fork boundaries")
    hw = c["hardware"]
    devices = hw["trainer_physical_gpus"] + hw["rollout_physical_gpus"]
    require(hw["gpu_count"] == 4 and sorted(devices) == [0, 1, 2, 3], "four disjoint GPU slots required")
    require(hw["schedule"] == "split_2train_2rollout", "alternative scheduling needs measured protocol")
    require(c["resources"]["charge_allocated_idle_devices"] is True, "allocated cost must include idle time")
    ceiling = c["resources"]["project_gpu_hour_cap"]
    require(type(ceiling) in (int, float) and math.isfinite(ceiling) and ceiling > 0, "invalid project cap")
    run_cap = c["resources"]["run_gpu_hour_cap"]
    if run_cap is not None:
        require(type(run_cap) in (int, float) and math.isfinite(run_cap) and 0 < run_cap <= ceiling,
                "invalid run cap")
    if errors:
        raise ValueError("; ".join(errors))
    return c


def production_blockers(c):
    blockers = []
    for group, key in (("model", "writer_revision"), ("model", "reader_revision"),
                       ("tokenization", "reference_revision")):
        if not re.fullmatch(r"[0-9a-f]{40}", str(c[group][key])):
            blockers.append(f"pin {group}.{key} to an immutable 40-hex revision")
    if c["model"]["writer_revision"] != c["model"]["reader_revision"]:
        blockers.append("shared-engine writer/reader revisions differ")
    native = c["model"]["native_window"]
    if type(native) is not int or native < max(c["writer_budget"]["total"], c["reader_budget"]["total"]):
        blockers.append("verify model.native_window")
    if c["observation"]["manifest"] == "BUILD_ONCE_AND_FREEZE":
        blockers.append("freeze observation manifest")
    for field in ("runtime_lock", "data_manifest", "gate_receipts"):
        value = c["artifacts"][field]
        if not isinstance(value, str) or not (ROOT / value).is_file():
            blockers.append(f"supply artifacts.{field}")
    if c["resources"]["run_gpu_hour_cap"] is None:
        blockers.append("set a measured run GPU-hour cap")
    if c["initialization"]["use_shared_format_initialization"] and not c["initialization"]["writer_adapter_sha256"]:
        blockers.append("hash the shared format adapter")
    if not (ROOT / "src/memrl/runtime/launch.py").is_file():
        blockers.append("production launcher has not been implemented")
    # This helper deliberately never certifies launch eligibility from filenames.
    blockers.append("the production runtime must verify lock contents, gate evidence, lineage, and exact tokenizer budgets")
    return blockers


def resolve(profile_path, base_path=None):
    base = load_json(base_path or ROOT / "configs/base.json")
    return validate(merge_strict(base, load_json(profile_path)))
