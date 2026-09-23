"""Production configuration resolution and launch certification."""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import copy

from memrl.contracts.common import commit_hex, sha256_hex
from memrl.contracts.types import LaunchReceipt
from memrl.io import fingerprint, load_json, sha256_text

ROOT = Path(__file__).resolve().parents[2]
FINAL_MARKERS = ("longmemeval", "locomo")
LOCK_KEYS = frozenset({
    "schema_version", "status", "python_version", "container_image_digest",
    "driver_version", "cuda_runtime", "pytorch_version", "vllm_version",
    "transformers_version", "peft_version", "tokenizers_version",
    "flash_attention_version", "dependency_lock_sha256", "base_model_revision",
    "tokenizer_revision", "chat_template_sha256", "gpu_uuids", "gpu_memory_bytes",
    "topology_artifact", "api_probe_artifact", "logprob_parity_artifact",
    "parity_tolerance", "reader_adapter_is_none_verified",
})
_VERSION_KEYS = (
    "python_version", "driver_version", "cuda_runtime", "pytorch_version",
    "vllm_version", "transformers_version", "peft_version", "tokenizers_version",
)


def _planning():
    path = ROOT / "scripts" / "config_contract.py"
    spec = importlib.util.spec_from_file_location("memrl_planning_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("planning config contract is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_runtime_lock(lock: dict[str, Any]) -> dict[str, Any]:
    """Validate a runtime lock without merging its fields into the scientific config."""
    if not isinstance(lock, dict):
        raise ValueError("runtime lock must be an object")
    extra = set(lock) - LOCK_KEYS
    missing = LOCK_KEYS - set(lock)
    if extra:
        raise ValueError(f"unknown runtime lock key: {sorted(extra)[0]}")
    if missing:
        raise ValueError(f"runtime lock missing {sorted(missing)[0]}")
    if lock["schema_version"] != 3:
        raise ValueError("runtime lock schema_version must be 3")
    if lock["status"] not in {"unverified_template", "verified"}:
        raise ValueError("runtime lock status must be unverified_template or verified")
    if lock["status"] == "unverified_template":
        return lock
    commit_hex(lock["base_model_revision"], "base_model_revision")
    commit_hex(lock["tokenizer_revision"], "tokenizer_revision")
    sha256_hex(lock["chat_template_sha256"], "chat_template_sha256")
    for key in _VERSION_KEYS:
        if type(lock[key]) is not str or not lock[key].strip():
            raise ValueError(f"verified runtime lock requires {key}")
    if type(lock["gpu_uuids"]) is not list or not lock["gpu_uuids"]:
        raise ValueError("verified runtime lock requires gpu_uuids")
    if lock["reader_adapter_is_none_verified"] is not True:
        raise ValueError("reader adapter isolation is not verified")
    if type(lock["logprob_parity_artifact"]) is not str or not lock["logprob_parity_artifact"]:
        raise ValueError("logprob parity artifact is missing")
    return lock


def apply_runtime_lock(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Copy pinned revisions into config fields. Other lock keys stay out of the config."""
    checked = validate_runtime_lock(lock)
    if checked["status"] != "verified":
        return config
    result = copy.deepcopy(config)
    revision = checked["base_model_revision"]
    result["model"]["writer_revision"] = revision
    result["model"]["reader_revision"] = revision
    result["tokenization"]["reference_revision"] = checked["tokenizer_revision"]
    return result


def resolve(profile_path: str | Path, base_path: str | Path | None = None,
            runtime_lock: dict[str, Any] | None = None,
            cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge base, profile, then CLI overrides. A runtime lock is applied, not merged."""
    planning = _planning()
    base = planning.load_json(base_path or ROOT / "configs" / "base.json")
    merged = planning.merge_strict(base, planning.load_json(profile_path))
    if runtime_lock:
        merged = apply_runtime_lock(merged, runtime_lock)
    if cli_overrides:
        merged = planning.merge_strict(merged, cli_overrides)
    return planning.validate(merged)


def resolved_hash(config: dict[str, Any]) -> str:
    return fingerprint(config)


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("code commit is unresolved") from exc
    if len(commit) != 40:
        raise RuntimeError("code commit is unresolved")
    return commit


def _dirty_hash() -> str:
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("dirty diff is unresolved") from exc
    return sha256_text(diff)


def _require_passed(gates: dict[str, Any], names: tuple[str, ...]) -> None:
    table = gates.get("gates", gates)
    for name in names:
        row = table.get(name)
        if not isinstance(row, dict) or row.get("status") != "passed":
            raise ValueError(f"gate {name} is not passed")


def validate_for_launch(config: dict[str, Any], lock: dict[str, Any],
                        manifests: dict[str, Any], gate_records: dict[str, Any],
                        *, kind: str = "train") -> LaunchReceipt:
    """Certify a launch. Raises on any missing pin, gate, or budget."""
    planning = _planning()
    planning.validate(config)
    checked = validate_runtime_lock(lock)
    if checked["status"] != "verified":
        raise ValueError("runtime lock is an unverified template")
    if checked["base_model_revision"] != config["model"]["writer_revision"]:
        raise ValueError("lock base revision does not match writer revision")
    if checked["tokenizer_revision"] != config["tokenization"]["reference_revision"]:
        raise ValueError("lock tokenizer revision does not match reference tokenizer")
    blockers = planning.production_blockers(config)
    real = [item for item in blockers if "production runtime must verify" not in item]
    if real:
        raise ValueError("; ".join(real))
    if kind == "train":
        _require_passed(gate_records, ("G00", "G01", "G02", "G03", "G04", "G05", "G06"))
    elif kind == "evaluate":
        _require_passed(gate_records, ("G00", "G01", "G02", "G03"))
    else:
        raise ValueError("unknown launch kind")
    manifest_text = manifests.get("canonical_text")
    if not isinstance(manifest_text, str):
        raise ValueError("data manifest canonical text is required")
    if config["data"]["allow_external_test_for_selection"]:
        raise ValueError("final-benchmark path in the checkpoint selector")
    joined = " ".join(str(manifests.get("paths", ""))).lower()
    if kind == "train" and any(marker in joined for marker in FINAL_MARKERS):
        raise ValueError("final-benchmark path in the checkpoint selector")
    cap = config["resources"]["run_gpu_hour_cap"]
    used = float(manifests.get("gpu_hours_already_allocated", 0.0))
    if used < 0 or used > cap:
        raise ValueError("GPU-hour budget exceeded or invalid")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return LaunchReceipt(
        protocol_id=config["run"]["protocol"],
        config_sha256=resolved_hash(config),
        code_commit=_git_commit(),
        code_dirty_diff_sha256=_dirty_hash(),
        environment_lock_sha256=fingerprint(lock),
        data_manifest_sha256=sha256_text(manifest_text),
        gate_record_sha256=fingerprint(gate_records),
        resource_lease_id=str(manifests["resource_lease_id"]),
        created_at_utc=now,
    )


def assert_not_final_selector(path_text: str, split: str) -> None:
    if split != "development":
        raise ValueError("final-benchmark path in the checkpoint selector")
    lowered = path_text.lower()
    if any(marker in lowered for marker in FINAL_MARKERS):
        raise ValueError("final-benchmark path in the checkpoint selector")


def load_resolved(path: str | Path) -> dict[str, Any]:
    return _planning().validate(load_json(path))
