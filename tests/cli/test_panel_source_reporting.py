# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say when a declared panel was discarded (#16).

The reporting half of persona sourcing, and the sixth member of #40's first exit
criterion to be closed. The class test is **distinguishability** (#74's close-out
wording), not the presence of a word: before this, a repo whose ``CLAUDE.md``
declared a panel that parsed to nobody produced stdout **and** a ``--- JSON ---``
artefact byte-identical to a repo that declared no panel at all, and both said
``7 personas from default``. ``source: "default"`` informs only a reader who
already knows what they declared, which is exactly what an artefact read cold
does not.

Two shapes, and the split is deliberate:

* the **console** warning is silent unless something was discarded, and lands
  before the first epoch — the one moment an operator can still abort. A repo
  that declared nothing asked for nothing and must read clean, because a rule
  that fires on a healthy run is the mirror image of the defect;
* the **artefact** key is unconditional, because that is where a run read back
  cold does the distinguishing, and an absent key makes no claim.

Driven through ``kuang.cli.main`` end to end with nothing spawned.
"""

from __future__ import annotations

import json

import pytest

from kuang.cli import main

_GOOD_EXPERTS = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

# `###` instead of `##`, and a colon instead of " — ". Two plausible slips, and
# the panel is gone with no signal.
_EXPERTS_THAT_PARSE_TO_ZERO = """\
# Resident Experts

### The Analyst: correctness

Does the change compute the right thing?
"""


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", "--dry-run", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


# --- the class test ----------------------------------------------------------

def test_a_discarded_panel_is_distinguishable_from_none_in_the_run_output(
        changed_repo, capsys):
    """The defect, stated as the criterion states it.

    Two runs, identical but for a ``CLAUDE.md`` that declares a panel and yields
    nobody. On main their stdout and artefact were byte-identical.
    """
    _run(changed_repo)
    clean = capsys.readouterr().out

    (changed_repo / "CLAUDE.md").write_text(_EXPERTS_THAT_PARSE_TO_ZERO)
    _run(changed_repo)
    degraded = capsys.readouterr().out

    assert clean != degraded
    assert _artefact(clean)["panel"] != _artefact(degraded)["panel"]


def test_the_discard_reaches_both_the_console_and_the_artefact(
        changed_repo, capsys):
    """Named, with its cause, in the stream an operator watches AND the one
    ``--prior-findings`` consumes."""
    (changed_repo / "CLAUDE.md").write_text(_EXPERTS_THAT_PARSE_TO_ZERO)
    _run(changed_repo)
    out = capsys.readouterr().out

    assert "1 declared panel(s) DISCARDED" in out
    assert "CLAUDE.md: it named no persona" in out
    assert _artefact(out)["panel"]["discarded"] == [
        {"origin": "CLAUDE.md", "reason": "empty", "detail": ""}]
    assert _artefact(out)["panel"]["source"] == "default"


def test_the_warning_lands_before_anything_is_spent(changed_repo, capsys):
    """It is worth nothing after the panel has been paid for.

    Asserted by position, not by claim: the warning must precede every line the
    run emits about reviewing.
    """
    (changed_repo / "CLAUDE.md").write_text(_EXPERTS_THAT_PARSE_TO_ZERO)
    _run(changed_repo)
    out = capsys.readouterr().out

    assert out.index("DISCARDED") < out.index("review surface:")
    assert out.index("DISCARDED") < out.index("--- JSON ---")


# --- the mirror image --------------------------------------------------------

@pytest.mark.parametrize("claude_md", [None, _GOOD_EXPERTS],
                         ids=["no-CLAUDE.md-at-all", "a-panel-that-worked"])
def test_a_run_that_declared_nothing_is_not_warned_at(
        changed_repo, capsys, claude_md):
    """A rule that fires on a healthy run is the defect again, inverted.

    Paired with the tests above, which forbid passing this one vacuously by
    reporting nothing anywhere.
    """
    if claude_md:
        (changed_repo / "CLAUDE.md").write_text(claude_md)
    _run(changed_repo)
    out = capsys.readouterr().out

    assert "DISCARDED" not in out
    assert "WARNING" not in out
    assert _artefact(out)["panel"]["discarded"] == []


def test_a_CLAUDE_md_that_declares_no_panel_is_not_a_discard(changed_repo, capsys):
    """The boundary, and the case the whole rule turns on.

    A ``CLAUDE.md`` exists for reasons other than declaring a panel; we read one
    section out of it. A file carrying no ``Resident Experts`` heading declared
    nothing **to us**, so the run is clean — this is the option the fix rejected,
    and it is asserted rather than left to the docstring.
    """
    (changed_repo / "CLAUDE.md").write_text("# A repo\n\nHouse rules, no panel.\n")
    _run(changed_repo)
    out = capsys.readouterr().out

    assert _artefact(out)["panel"]["discarded"] == []
    assert "DISCARDED" not in out


# --- --panel -----------------------------------------------------------------

def test_an_explicit_panel_is_used_and_named(changed_repo, capsys):
    """Precedence over every other tier, and the file is echoed, not just the flag."""
    (changed_repo / "CLAUDE.md").write_text(_GOOD_EXPERTS)
    named = changed_repo / "docs" / "experts.md"
    named.parent.mkdir()
    named.write_text("# Resident Experts\n\n## Auditor — Constraints\n\nRules.\n")

    assert _run(changed_repo, "--panel", str(named)) == 0
    out = capsys.readouterr().out

    assert f"personas from --panel {named}:" in out
    assert "Auditor" in out and "Analyst" not in out
    payload = _artefact(out)["panel"]
    assert payload["source"] == "--panel"
    assert payload["path"] == str(named)
    assert payload["discarded"] == []


def test_an_explicit_panel_that_cannot_be_used_refuses_without_a_traceback(
        changed_repo, capsys):
    """An explicit instruction is never silently overridden.

    Consistent with the existing operator-error path rather than a third
    behaviour for the class (#59), and nothing is spent: it fires before the
    first epoch, so there is no artefact and no roster line.
    """
    assert _run(changed_repo, "--panel", str(changed_repo / "nope.md")) == 2
    captured = capsys.readouterr()

    assert captured.err.startswith("error: --panel ")
    assert "the file does not exist" in captured.err
    assert "Traceback" not in captured.err
    assert "--- JSON ---" not in captured.out
    assert "[panel]" not in captured.out


def test_the_missing_yaml_extra_names_the_extra(changed_repo, capsys, monkeypatch):
    """The sharpest cause: a VALID panel silently ignored on a default install.

    ``pyyaml`` is an optional extra a plain ``pip install`` does not bring, and it
    is absent from this repo's own venv. Forced here rather than assumed, so the
    test asserts the branch and not the machine.
    """
    import sys
    monkeypatch.setitem(sys.modules, "yaml", None)
    (changed_repo / "panel.yaml").write_text("personas:\n  - name: A\n")

    _run(changed_repo)
    out = capsys.readouterr().out

    assert "panel.yaml: reading a YAML panel needs the optional extra" in out
    assert "pip install 'kuang[yaml]'" in out
    assert _artefact(out)["panel"]["discarded"][0]["reason"] == "unsupported"


def test_every_discarded_declaration_is_counted_and_named(
        changed_repo, capsys, monkeypatch):
    """Two at once, because a denominator nobody pins is a denominator nobody checks.

    A single-discard run cannot tell a real count from a hard-coded 1 — the gap a
    mutation found here, and the same one #82 found in its own first matrix.
    """
    import sys
    monkeypatch.setitem(sys.modules, "yaml", None)
    (changed_repo / "CLAUDE.md").write_text(_EXPERTS_THAT_PARSE_TO_ZERO)
    (changed_repo / "panel.yaml").write_text("personas:\n  - name: A\n")

    _run(changed_repo)
    out = capsys.readouterr().out

    assert "2 declared panel(s) DISCARDED" in out
    assert "[panel]   CLAUDE.md:" in out
    assert "[panel]   panel.yaml:" in out
    assert [d["origin"] for d in _artefact(out)["panel"]["discarded"]] == [
        "CLAUDE.md", "panel.yaml"]


def test_the_runtime_own_words_reach_the_console_not_only_the_artefact(
        changed_repo, capsys, monkeypatch):
    """Record and report, never discard — including the diagnosis itself.

    The detail is what the runtime actually said. Before #16 the only cause that
    said anything said it on stderr; a fix that kept the categorical reason and
    dropped the words would be the same loss in a new place.

    The two streams are pinned to each other rather than to a literal message,
    because the wording is the *runtime's* and can change under us — the decay
    that killed #77's capture while its shape held.
    """
    import sys
    monkeypatch.setitem(sys.modules, "yaml", None)
    (changed_repo / "panel.yaml").write_text("personas:\n  - name: A\n")

    _run(changed_repo)
    out = capsys.readouterr().out
    detail = _artefact(out)["panel"]["discarded"][0]["detail"]

    assert detail, "the runtime said something and the record must keep it"
    assert f"— {detail}" in out


def test_the_panel_path_is_null_when_the_operator_named_nothing(
        changed_repo, capsys):
    """Three-state, not two: ``null`` is "no file was named", a fact rather than
    a decline, and it is what makes the key readable on every run."""
    _run(changed_repo)

    assert _artefact(capsys.readouterr().out)["panel"]["path"] is None
