import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WireSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jsonschema
        cls.jsonschema = jsonschema

    def _schema(self, name):
        return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))

    def _example(self, name):
        return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))

    def test_valid_observation_and_weighted_call(self):
        self.jsonschema.validate(
            self._example("valid_observation.json"),
            self._schema("public_observation.schema.json"),
        )
        self.jsonschema.validate(
            self._example("valid_weighted_call.json"),
            self._schema("weighted_call.schema.json"),
        )

    def test_invalid_examples_fail(self):
        with self.assertRaises(self.jsonschema.ValidationError):
            self.jsonschema.validate(
                self._example("invalid_observation_extra.json"),
                self._schema("public_observation.schema.json"),
            )
        with self.assertRaises(self.jsonschema.ValidationError):
            self.jsonschema.validate(
                self._example("invalid_weighted_call_prefix_branch.json"),
                self._schema("weighted_call.schema.json"),
            )
