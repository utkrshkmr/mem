"""Check package structure, draft profiles and source syntax. Not a GPU gate."""
import ast
from pathlib import Path
import re
import sys

from config_contract import ROOT, load_json, resolve

REQUIRED = [
    "README.md", "AGENTS.md", "MEMRL_SPEC.md", "pyproject.toml",
    "docs/CONTINUATION_METHOD.md", "docs/INTERFACES.md", "docs/PARALLEL_BUILD.md",
    "docs/ACCEPTANCE_GATES.md", "docs/EXPERIMENTS.md", "docs/NOVELTY_AND_VENUE.md",
    "docs/FOUR_H100_RUNBOOK.md", "docs/RUNTIME_WIRING.md", "docs/REPRODUCIBILITY.md",
    "configs/base.json", "configs/experiment_registry.json", "prompts/COORDINATOR.md",
    "locks/runtime_lock.template.json", "records/gates.json", "paper/CLAIMS.md",
]


def main():
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append("missing required file: " + relative)
    files = [p for p in ROOT.rglob("*") if p.is_file() and not any(
        part in (".git", ".venv", "__pycache__", "build", "runs", "data") for part in p.relative_to(ROOT).parts)]
    for p in files:
        try:
            if p.suffix == ".json":
                load_json(p)
            elif p.suffix == ".py":
                ast.parse(p.read_text(), filename=str(p))
            elif p.suffix in (".md", ".mdc"):
                text = p.read_text()
                # Fence stack accepts either standard Markdown delimiter.
                fence = None
                for line in text.splitlines():
                    match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
                    if match:
                        mark, rest = match.groups()
                        if fence is None:
                            fence = mark
                        elif mark[0] == fence[0] and len(mark) >= len(fence) and not rest.strip():
                            fence = None
                if fence:
                    errors.append(f"unclosed Markdown fence: {p.relative_to(ROOT)}")
                for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", text):
                    if "://" in target or target.startswith(("#", "mailto:", "sandbox:")):
                        continue
                    path = target.split("#", 1)[0]
                    if path and not (p.parent / path).exists():
                        errors.append(f"missing local link: {p.relative_to(ROOT)} -> {path}")
        except (ValueError, SyntaxError, OSError) as exc:
            errors.append(f"{p.relative_to(ROOT)}: {exc}")
    for p in (ROOT / "configs/profiles").glob("*.json"):
        try:
            resolve(p)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"profile {p.name}: {exc}")
    tasks = list((ROOT / "prompts/agents").glob("*.md"))
    if len(tasks) != 7:
        errors.append("expected coordinator plus six agent task packets")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Handoff validation passed: {len(files)} files; 6 arm profiles; 7 task packets.")
    print("This checks structure and CPU planning contracts only; no GPU or research result is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
