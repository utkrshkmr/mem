from dataclasses import FrozenInstanceError
import json
import unittest

from memrl_contract.contracts import (
    ModelLock, PrivateHistory, PrivateQuestion, PublicChunk, PublicHistory,
    canonical_json, public_projection, serialize_public,
)


class PublicBoundaryTests(unittest.TestCase):
    def test_projection_is_explicit_and_excludes_all_secrets(self):
        public = PublicHistory("opaque-history", (PublicChunk("chunk-0", 0, "observed text"),))
        private = PrivateHistory(
            public, (PrivateQuestion("secret-question-id", "SECRET_QUERY", "SECRET_GOLD", "SECRET_COHORT"),),
            "SECRET_WORLD_DIGEST",
        )
        projected = public_projection(private)
        encoded = serialize_public(projected)
        self.assertEqual(projected, public)
        self.assertIsNot(projected, public)
        self.assertIsNot(projected.chunks[0], public.chunks[0])
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("secret-question-id", encoded)
        self.assertEqual(set(json.loads(encoded)), {"history_id", "chunks"})
        with self.assertRaises(ValueError):
            serialize_public(private)

    def test_public_data_is_immutable(self):
        chunk = PublicChunk("chunk-0", 0, "source")
        with self.assertRaises(FrozenInstanceError):
            chunk.text = "edited"
        with self.assertRaises(ValueError):
            PublicHistory("id", [chunk])

    def test_sequence_and_utf8_validation(self):
        with self.assertRaises(ValueError):
            PublicHistory("id", (PublicChunk("a", 1, "bad ordinal"),))
        with self.assertRaises(ValueError):
            PublicHistory("id", (PublicChunk("a", 0, "x"), PublicChunk("a", 1, "y")))
        with self.assertRaises(ValueError):
            PublicChunk("id", True, "bool is not an ordinal")
        with self.assertRaises(ValueError):
            PublicChunk("id", 0, "\ud800")

    def test_canonical_manifest_rejects_nonfinite_json(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')
        with self.assertRaises(ValueError):
            canonical_json({"reward": float("nan")})


class ModelLockTests(unittest.TestCase):
    def lock(self, **changes):
        values = dict(
            repository="example/model", revision="a" * 40,
            tokenizer_repository="example/model", tokenizer_revision="b" * 40,
            chat_template_sha256="c" * 64,
        )
        values.update(changes)
        return ModelLock(**values)

    def test_resolved_identity(self):
        self.assertEqual(self.lock().dtype, "bfloat16")

    def test_floating_or_placeholder_revisions_fail(self):
        for bad in ("main", "latest", "RESOLVE_AND_PIN", "a" * 39, "z" * 40):
            with self.subTest(revision=bad), self.assertRaises(ValueError):
                self.lock(revision=bad)
        with self.assertRaises(ValueError):
            self.lock(tokenizer_revision="main")
        with self.assertRaises(ValueError):
            self.lock(chat_template_sha256="unknown")


if __name__ == "__main__":
    unittest.main()

