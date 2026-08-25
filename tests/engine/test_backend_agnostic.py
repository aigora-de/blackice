# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The engine is backend-agnostic — asserted, not merely claimed.

``CLAUDE.md`` mandates the three-concern separation: engine, backend, entry
point. Until #19 that claim rested entirely on nobody having broken it, which is
not a property, it is a run of luck. These tests read the import graph out of the
source and hold the dependency to one direction: backends and the CLI may import
the engine; the engine may import neither.

**What this does not prove.** It is a *static* check over ``import`` statements,
including ones nested inside functions. A runtime
``importlib.import_module("blackice.backends…")`` would evade it, as would any
other dynamically-constructed import. That is a deliberate limit: the check is
cheap, deterministic and reads the same tree a human would, and the failure mode
it guards is an ordinary import added without thinking, not a deliberate
circumvention.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / "blackice"


def _imports(path: Path) -> set[str]:
    """Every module ``path`` imports, as a dotted name, relative ones resolved."""
    package = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # a relative import
                base = package[: len(package) - (node.level - 1)]
                names.add(".".join(base + (node.module.split(".") if node.module else [])))
            elif node.module:
                names.add(node.module)
    return names


def _modules(*parts: str) -> list[Path]:
    files = sorted(PACKAGE.joinpath(*parts).rglob("*.py"))
    assert files, f"no modules found under {'/'.join(parts)} — the check is vacuous"
    return files


def _offenders(files: list[Path], forbidden: tuple[str, ...]) -> dict[str, set[str]]:
    out = {}
    for path in files:
        bad = {m for m in _imports(path)
               if any(m == f or m.startswith(f + ".") for f in forbidden)}
        if bad:
            out[str(path.relative_to(PACKAGE.parent))] = bad
    return out


def test_the_engine_imports_nothing_from_a_backend():
    """The seam CLAUDE.md mandates: the engine knows of no particular runtime."""
    assert _offenders(_modules("engine"),
                      ("blackice.backends", "claude_code_backend")) == {}


def test_the_engine_imports_nothing_from_the_entry_point():
    assert _offenders(_modules("engine"), ("blackice.cli",)) == {}


def test_no_backend_imports_the_entry_point():
    """CLI wiring flows into a backend, never out of one."""
    assert _offenders(_modules("backends"), ("blackice.cli",)) == {}


def test_the_engine_is_importable_without_any_backend(monkeypatch):
    """Not just unimported — unimportABLE would be a lie if it still loaded one."""
    import sys

    for name in [m for m in sys.modules if m.startswith("blackice")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "blackice.backends", None)

    import blackice.engine  # noqa: F401  — must not touch blackice.backends

    assert all(not m.startswith("blackice.backends.")
               for m in sys.modules if m.startswith("blackice"))


def test_the_entry_point_is_the_only_module_that_knows_both():
    """The converse: the seam exists because the CLI spans it, not by accident."""
    cli = set().union(*(_imports(p) for p in _modules("cli")))

    assert any(m.startswith("blackice.engine") for m in cli)
    assert any(m.startswith("blackice.backends") for m in cli)


@pytest.mark.parametrize("relative, expected", [
    ("from .findings import Finding", "blackice.engine.findings"),
    ("from ..report import ledger_line", "blackice.report"),
    ("from . import findings", "blackice.engine"),
])
def test_the_checker_resolves_relative_imports(tmp_path, relative, expected, monkeypatch):
    """The check is only worth its green if it can see a relative import."""
    module = PACKAGE / "engine" / "_probe_for_tests.py"
    module.write_text(relative + "\n")
    try:
        assert expected in _imports(module)
    finally:
        module.unlink()
