import json
from pathlib import Path
import unittest

from memrl.config import apply_runtime_lock, resolve, validate_runtime_lock
from memrl.io import load_json

ROOT = Path(__file__).resolve().parents[1]


def _template():
    return load_json(ROOT / "locks" / "runtime_lock.template.json")


class RuntimeLockTests(unittest.TestCase):
    def test_lock_keys_are_not_merged_into_the_scientific_config(self):
        profile = ROOT / "configs" / "profiles" / "F03_fork.json"
        lock = _template()
        resolved = resolve(profile, runtime_lock=lock)
        self.assertNotIn("python_version", resolved)
        self.assertEqual(resolved["model"]["writer_revision"], "RESOLVE_AND_PIN")

    def test_unknown_lock_key_fails(self):
        lock = _template()
        lock["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "unknown runtime lock key"):
            validate_runtime_lock(lock)

    def test_verified_lock_copies_revisions_only(self):
        lock = _template()
        lock["status"] = "verified"
        revision = "a" * 40
        lock["base_model_revision"] = revision
        lock["tokenizer_revision"] = "b" * 40
        lock["chat_template_sha256"] = "c" * 64
        for key in (
            "python_version", "driver_version", "cuda_runtime", "pytorch_version",
            "vllm_version", "transformers_version", "peft_version", "tokenizers_version",
        ):
            lock[key] = "1"
        lock["gpu_uuids"] = ["GPU-1"]
        lock["reader_adapter_is_none_verified"] = True
        lock["logprob_parity_artifact"] = "reports/api_probe/hf_logprob.json"
        base = load_json(ROOT / "configs" / "base.json")
        applied = apply_runtime_lock(base, lock)
        self.assertEqual(applied["model"]["writer_revision"], revision)
        self.assertEqual(applied["model"]["reader_revision"], revision)
        self.assertEqual(applied["tokenization"]["reference_revision"], "b" * 40)
        self.assertNotIn("vllm_version", applied)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            bad = json.loads(json.dumps(lock))
            bad["chat_template_sha256"] = "0" * 64
            validate_runtime_lock(bad)
