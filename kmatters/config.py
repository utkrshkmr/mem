"""Configuration: YAML files merged left to right, then `a.b=c` overrides, validated by pydantic.

Unknown keys are errors. The canonical JSON dump of the resolved config is hashed
(sha256) into `config_hash`, which is stored with every output (PLAN.md §7).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCfg(_Strict):
    repo: str
    local_dir: str
    dtype: Literal["bfloat16"]
    attn_implementation: str


class LoraCfg(_Strict):
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    init_seed_offset: int

    @field_validator("dropout")
    @classmethod
    def _dropout_zero(cls, v):
        if v != 0.0:
            raise ValueError("lora.dropout must be 0.0: dropout would make the trained policy "
                             "differ from the sampling policy")
        return v


class VllmCfg(_Strict):
    kv_cache_memory_bytes: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_prefix_caching: bool
    enforce_eager: bool
    max_loras: int
    max_lora_rank: int
    max_cpu_loras: int
    per_request_seeds: bool


class EnvCfg(_Strict):
    level: int
    L: int
    U: int
    B: int

    @field_validator("level")
    @classmethod
    def _level_range(cls, v):
        if v not in (1, 2, 3, 4, 5):
            raise ValueError("env.level must be in 1..5")
        return v


class SamplingCfg(_Strict):
    writer_temperature: float
    reader_max_tokens: int

    @field_validator("writer_temperature")
    @classmethod
    def _on_policy(cls, v):
        if v != 1.0:
            raise ValueError("sampling.writer_temperature must be 1.0 (exact on-policy sampling)")
        return v


class EstimatorCfg(_Strict):
    baseline: Literal["loo_prefix", "tree"]
    T_norm: float


class TrainCfg(_Strict):
    F: int
    K: int
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip: float | None
    max_tokens_per_microbatch: int
    iterations: int
    seed: int
    executor: Literal["async_ready", "barrier"]
    eval_every: int
    ckpt_every: int
    keep_optimizer_ckpts: int

    @property
    def N(self) -> int:
        return self.F // self.K


class EvalSetCfg(_Strict):
    n_prefixes: int
    futures_per_prefix: int
    eval_seed: int


class EvalCfg(_Strict):
    small: EvalSetCfg
    large: EvalSetCfg


class ProfileCfg(_Strict):
    reps: int
    warmup: int
    lr: float


class VarianceCfg(_Strict):
    N0: int
    K0: int


class PathsCfg(_Strict):
    runs: str
    profiles: str
    variance: str
    reports: str


class Config(_Strict):
    model: ModelCfg
    lora: LoraCfg
    vllm: VllmCfg
    env: EnvCfg
    sampling: SamplingCfg
    estimator: EstimatorCfg
    train: TrainCfg
    eval: EvalCfg
    profile: ProfileCfg
    variance: VarianceCfg
    paths: PathsCfg

    @model_validator(mode="after")
    def _check(self):
        if self.train.K < 1 or self.train.F % self.train.K != 0:
            raise ValueError(f"N = F / K must be an integer (F={self.train.F}, K={self.train.K})")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def run_id(self, tag: str) -> str:
        return f"{tag}_K{self.train.K}_N{self.train.N}_s{self.train.seed}_{self.config_hash[:8]}"


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_override(d: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like a.b=c, got {item!r}")
    path, raw = item.split("=", 1)
    keys = path.strip().split(".")
    node = d
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            raise ValueError(f"unknown config section in override {item!r}")
        node = node[k]
    if keys[-1] not in node:
        raise ValueError(f"unknown config key in override {item!r}")
    node[keys[-1]] = yaml.safe_load(raw)


def load(paths=("configs/base.yaml",), overrides=()) -> Config:
    """Merge YAML files left to right, apply `a.b=c` overrides, validate."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict = {}
    for p in paths:
        merged = _deep_merge(merged, yaml.safe_load(Path(p).read_text()) or {})
    for item in overrides:
        _apply_override(merged, item)
    return Config.model_validate(merged)


def save_resolved(cfg: Config, out_dir: str | Path) -> None:
    """Write the canonical resolved config and its hash next to an output."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(json.loads(cfg.canonical_json()), indent=2, sort_keys=True))
    (out / "config_hash").write_text(cfg.config_hash + "\n")
