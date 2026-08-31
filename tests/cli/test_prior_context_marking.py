# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A degraded call's finding must not become the next panel's prior context (#71).

End to end through ``kuang.cli.main``, subprocess boundary stubbed, so the prompt
a persona is actually handed in epoch 2 is asserted rather than reasoned about.

Two ends of one round trip. Cross-*epoch*: ``on_epoch`` renders the ledger into
``prior_summary`` and ``build_prompt`` injects it from epoch 2. Cross-*run*: the
CLI writes the same provenance into the artefact, so a run seeded from a
contaminated artefact with ``--prior-findings`` (#13) inherits the marks rather
than reading clean.

Nothing is excluded — the marked lines are still there. What changes is that the
panel is told which lines this run knows were not grounded in the source, and what
to do about each.
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

_HEALTHY_TURNS = 7
_NO_TOOL_TURNS = 1


def _finding(title, severity="BLOCKER", line=1):
    return {"title": title, "severity": severity, "claim_class": "logic",
            "file": "a.py", "line": line, "evidence": "read it"}


def _contract(findings=(), verdict="NO", turns=_HEALTHY_TURNS) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0,
                      num_turns=turns)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies, *, default=None) -> list[str]:
    """Stub the call boundary and return the list prompts accumulate into."""
    default = default if default is not None else _contract()
    prompts: list[str] = []

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        prompts.append(prompt)
        for name, reply in replies.items():
            if f"You are {name} " in mandate:
                return reply
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)
    return prompts


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


# --- the artefact, which is also the cross-run seed ---------------------------

def test_the_artefact_records_which_findings_were_ungrounded(
        sourced_repo, capsys, monkeypatch):
    """Auditable rather than asserted, and it is what a later run is seeded from."""
    _stub(monkeypatch, {
        "Analyst": _contract([_finding("guessed at it", line=1)],
                             turns=_NO_TOOL_TURNS),
        "Critic": _contract([_finding("read it", line=40)]),
    }, default=_contract(verdict="YES"))
    _run(sourced_repo, "--max-epochs", "1")

    by_title = {f["title"]: f["ungrounded"]
                for f in _artefact(capsys.readouterr().out)["findings"]}

    assert by_title == {"guessed at it": True, "read it": False}


def test_a_healthy_run_records_no_contamination(sourced_repo, capsys, monkeypatch):
    """The mirror image: a rule that fires on a healthy run is the defect again."""
    _stub(monkeypatch, {"Analyst": _contract([_finding("read it")])},
          default=_contract(verdict="YES"))
    _run(sourced_repo, "--max-epochs", "1")
    payload = _artefact(capsys.readouterr().out)

    assert payload["findings"], "the run must actually have produced a finding"
    assert not any(f["ungrounded"] or f["about_run"] for f in payload["findings"])


def test_a_dry_run_reports_no_contamination_it_did_not_suffer(
        sourced_repo, capsys, monkeypatch):
    """``checkpoint`` runs every epoch regardless of ``--dry-run`` (#67, #70, #69
    each had to answer this). Nothing was spawned, so the ledger is empty, there
    is nothing to mark, and no prior-context block is rendered at all."""
    _stub(monkeypatch, {})
    _run(sourced_repo, "--max-epochs", "2", "--dry-run")
    out = capsys.readouterr().out

    assert _artefact(out)["findings"] == []
    assert "PRIOR EPOCHS' FINDINGS" not in out
    assert "[ungrounded]" not in out
    assert "[about the run]" not in out


# --- what the next epoch's panel is actually handed ---------------------------

def test_the_next_epoch_is_told_which_prior_findings_were_ungrounded(
        sourced_repo, capsys, monkeypatch):
    """The whole point: one degraded call in epoch 1, three personas in epoch 2."""
    prompts = _stub(monkeypatch, {
        "Analyst": _contract([_finding("guessed at it")], turns=_NO_TOOL_TURNS),
    }, default=_contract(verdict="NO"))
    _run(sourced_repo, "--max-epochs", "2")
    capsys.readouterr()

    second_epoch = [p for p in prompts if "PRIOR EPOCHS' FINDINGS" in p]

    assert len(second_epoch) == 3, "every persona in epoch 2 gets prior context"
    for prompt in second_epoch:
        assert "guessed at it @ a.py:1 [ungrounded]" in prompt
        assert "check it against the source" in prompt


def test_a_seeded_run_inherits_the_mark_from_the_artefact(
        sourced_repo, tmp_path, capsys, monkeypatch):
    """Cross-run (#13): a contaminated artefact must not seed a later run clean."""
    seed = tmp_path / "prior.json"
    seed.write_text(json.dumps({"findings": [
        {"persona": "Analyst", "severity": "BLOCKER", "title": "guessed at it",
         "file": "a.py", "line": 1, "open": True, "about_run": False,
         "ungrounded": True}]}))
    prompts = _stub(monkeypatch, {}, default=_contract(verdict="YES"))
    _run(sourced_repo, "--max-epochs", "1", "--prior-findings", str(seed))
    capsys.readouterr()

    assert prompts, "the panel must have been spawned"
    assert all("guessed at it @ a.py:1 [ungrounded]" in p for p in prompts)
