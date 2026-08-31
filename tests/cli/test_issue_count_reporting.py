# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The headline issue count must not mix defects with instrument failures (#73).

`canonical issues: 2` counting one real BLOCKER in the code under review and one
`agent error:` from the panel reading it is the instrument adding itself to its
own results. #30 made the failure legible in the participation record; this stops
it being *counted* as a finding about the change.

Nothing is excluded to achieve it. Both kinds stay in the ledger and in the
artefact — #73's second acceptance criterion is that a failed persona's diagnosis
stays recoverable — and the count is reported split rather than the ledger being
filtered. Measured first: any exclusion rule keyed on `claim_class == "meta"`
drops a persona-declared ruin-class finding and unlatches the breaker, because
`claim_class` is model-controlled. See
`tests/engine/test_run_vs_change_findings.py` for that measurement.

End to end through `kuang.cli.main`, subprocess boundary stubbed.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

_AGENT_ERROR = CallResult("", 0, "agent error: error_max_turns: Reached maximum "
                                 "number of turns (30)")


def _finding(title, severity="BLOCKER", claim_class="logic", line=1):
    return {"title": title, "severity": severity, "claim_class": claim_class,
            "file": "a.py", "line": line, "evidence": "read it"}


def _contract(findings=(), verdict="NO") -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


# A conservative grouping that merges nothing: one cluster per finding.
def _grouping(n) -> CallResult:
    body = json.dumps({"clusters": [[i] for i in range(n)]})
    return CallResult(f"Grouped.\n\n```json\n{body}\n```\n", 0)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies, *, default=None, cluster=None) -> None:
    default = default if default is not None else _contract()

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "synthesiser" in mandate:
            return cluster if cluster is not None else _grouping(2)
        for name, reply in replies.items():
            if f"You are {name} " in mandate:
                return reply
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _headline(out: str) -> str:
    return next(ln for ln in out.splitlines() if ln.startswith("canonical issues:"))


# --- the defect ---------------------------------------------------------------

def test_the_headline_does_not_mix_defects_with_instrument_failures(
        sourced_repo, capsys, monkeypatch):
    """One real BLOCKER and one failed persona are not two canonical issues."""
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")]),
                        "Critic": _AGENT_ERROR}, default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    out = capsys.readouterr().out

    assert "1 in the change" in _headline(out), _headline(out)
    assert "1 about the run" in _headline(out)
    payload = _artefact(out)
    assert (payload["issues_in_change"], payload["issues_about_run"]) == (1, 1)


def test_the_artefact_marks_every_finding_and_every_cluster(
        sourced_repo, capsys, monkeypatch):
    """Auditable, not merely asserted — and nothing is dropped to achieve it."""
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")]),
                        "Critic": _AGENT_ERROR}, default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    payload = _artefact(capsys.readouterr().out)

    by_title = {f["title"]: f["about_run"] for f in payload["findings"]}
    assert by_title["off-by-one in the bound"] is False
    assert any(t.startswith("agent error:") and v for t, v in by_title.items()), \
        "the diagnosis must stay recoverable from the artefact (#73)"
    assert sorted(c["about_run"] for c in payload["clusters"]) == [False, True]


def test_the_cluster_line_says_which_ones_are_about_the_run(
        sourced_repo, capsys, monkeypatch):
    """The list under the headline, not only the headline itself."""
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")]),
                        "Critic": _AGENT_ERROR}, default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  [")]

    marked = [ln for ln in lines if "about the run" in ln]
    assert len(marked) == 1, lines
    assert "agent error:" in marked[0]
    assert not any("off-by-one" in ln for ln in marked)


# --- the mirror image ---------------------------------------------------------

def test_a_healthy_run_reports_nothing_about_the_run(sourced_repo, capsys,
                                                     monkeypatch):
    """A panel that did not fail must not be told it did.

    The count is always-on rather than an exception clause: "0 about the run" is
    the claim an operator needs, and an absent line cannot make it.
    """
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")])},
          default=_contract())
    _run(sourced_repo, "--semantic-dedup", "--max-epochs", "1")
    out = capsys.readouterr().out

    assert "0 about the run" in _headline(out), _headline(out)
    assert "1 in the change" in _headline(out)
    assert "about the run" not in "".join(
        ln for ln in out.splitlines() if ln.startswith("  ["))
    payload = _artefact(out)
    assert payload["issues_about_run"] == 0
    assert all(f["about_run"] is False for f in payload["findings"])


def test_a_persona_declared_meta_finding_is_counted_in_the_change(
        sourced_repo, capsys, monkeypatch):
    """End to end, the row that refutes a `claim_class` string match.

    A reviewer labelling a metaclass finding `meta` must not have it moved out of
    the change's count — nor, as the engine test pins, out of the ledger, where it
    would take the circuit-breaker with it.
    """
    _stub(monkeypatch,
          {"Sentinel": _contract([_finding("unbounded loss on retry", "UGLY", "meta")])},
          default=_contract())
    rc = _run(sourced_repo, "--semantic-dedup", )
    out = capsys.readouterr().out
    payload = _artefact(out)

    assert payload["halt_reason"] == "escalate_ugly" and rc == 3
    assert payload["issues_about_run"] == 0, "a persona's claim is about the change"
    assert payload["issues_in_change"] == 1
    assert all(f["about_run"] is False for f in payload["findings"])


# --- the property pair --------------------------------------------------------

def test_a_run_with_an_instrument_failure_cannot_be_read_as_one_without(
        sourced_repo, capsys, monkeypatch):
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")]),
                        "Critic": _AGENT_ERROR}, default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    degraded = capsys.readouterr().out
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")])},
          default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    clean = capsys.readouterr().out

    assert _headline(degraded) != _headline(clean)
    assert _artefact(degraded)["issues_about_run"] > _artefact(clean)["issues_about_run"]


def test_a_clean_run_is_not_reported_as_having_failed(sourced_repo, capsys,
                                                      monkeypatch):
    """The pair's other half: "degraded reads differently" passes trivially if
    every run reports a failure, so this forbids that."""
    _stub(monkeypatch, {"Analyst": _contract([_finding("off-by-one in the bound")])},
          default=_contract())
    _run(sourced_repo, "--semantic-dedup")
    payload = _artefact(capsys.readouterr().out)

    assert payload["issues_about_run"] == 0
    assert payload["issues_in_change"] == 1
