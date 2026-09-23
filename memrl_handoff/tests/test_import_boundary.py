import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("memrl.data.private", "memrl.eval", "memrl.pipeline")
GUARDED = ("memory", "env", "rl", "models", "runtime")


class ImportBoundaryTests(unittest.TestCase):
    def test_writer_packages_do_not_import_private_or_eval(self):
        base = ROOT / "src" / "memrl"
        for name in GUARDED:
            package = base / name
            if not package.exists():
                continue
            for path in package.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module]
                    else:
                        continue
                    for module in modules:
                        for forbidden in FORBIDDEN:
                            self.assertFalse(
                                module == forbidden or module.startswith(forbidden + "."),
                                f"{path} imports {module}",
                            )


if __name__ == "__main__":
    unittest.main()
