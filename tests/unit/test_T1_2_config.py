"""T1.2: config loading, overrides, validation and a stable hash."""
import subprocess
import sys

import pytest
from pydantic import ValidationError

from kmatters import config

BASE = str(config.__file__).rsplit("/kmatters/", 1)[0] + "/configs/base.yaml"


def test_T1_2_base_loads():
    cfg = config.load([BASE])
    assert cfg.train.F == 64 and cfg.train.K == 4 and cfg.train.N == 16
    assert cfg.lora.dropout == 0.0 and cfg.train.grad_clip is None
    assert cfg.eval.small.n_prefixes == 128


def test_T1_2_overrides_apply(tmp_path):
    extra = tmp_path / "extra.yaml"
    extra.write_text("env: {level: 2}\ntrain: {K: 8}\n")
    cfg = config.load([BASE, extra], ["train.lr=1e-4", "train.grad_clip=2.5", "eval.small.n_prefixes=96"])
    assert cfg.env.level == 2 and cfg.env.L == 16          # deep merge keeps sibling keys
    assert cfg.train.K == 8 and cfg.train.N == 8
    assert cfg.train.lr == 1e-4 and cfg.train.grad_clip == 2.5
    assert cfg.eval.small.n_prefixes == 96


@pytest.mark.parametrize("override", ["train.nope=1", "nosection.x=1", "train.F.x=1"])
def test_T1_2_unknown_key_raises(override):
    with pytest.raises(ValueError):
        config.load([BASE], [override])


def test_T1_2_unknown_yaml_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("train: {Kay: 4}\n")
    with pytest.raises(ValidationError):
        config.load([BASE, bad])


def test_T1_2_F_not_divisible_raises():
    with pytest.raises(ValidationError):
        config.load([BASE], ["train.K=3"])


def test_T1_2_dropout_nonzero_raises():
    with pytest.raises(ValidationError):
        config.load([BASE], ["lora.dropout=0.05"])


def test_T1_2_hash_stable_across_processes(record):
    code = ("from kmatters import config; import sys; "
            f"print(config.load([{BASE!r}], ['train.K=16']).config_hash)")
    h = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
         for _ in range(2)]
    here = config.load([BASE], ["train.K=16"]).config_hash
    assert h[0] == h[1] == here and len(here) == 64
    assert config.load([BASE]).config_hash != here
    cfg = config.load([BASE], ["train.K=16"])
    assert cfg.run_id("sweep") == f"sweep_K16_N4_s0_{here[:8]}"
    record("T1.2", config_hash_K16=here)
