# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A dead reduce step must not read as a working one (#30).

Measured on main: a clusterer that answered with a genuine grouping merging
nothing and a clusterer that never answered at all produced **byte-identical**
output bar the token count — the ``canonical issues: N (reduced from M)`` line,
the cluster list and the whole JSON object were the same. So "the ratio is the
tell" is refuted, not weak: an operator has no baseline for the ratio, and #65
has separately established that the token count is unreliable.

``session.reduce`` degrades to the identity reduce for five distinct reasons,
two of which are not degradations at all. The state is
recorded on the session — backend-local, exactly as ``tokens`` already is — so
the engine's ``Reduce`` seam keeps returning ``list[Cluster]`` and nothing else.

End to end through ``kuang.cli.main``, subprocess boundary stubbed.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.backends.claude_code.cluster import _CLUSTER_TEMPLATE
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


def _finding(title, line):
    return {"title": title, "severity": "BLOCKER", "claim_class": title,
            "file": "a.py", "line": line, "evidence": "read it"}


def _contract(findings) -> CallResult:
    body = json.dumps({"verdict": "NO", "findings": findings})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


# A genuine grouping that merges nothing: what a healthy conservative clusterer
# returns most of the time, and the variant the dead one was indistinguishable from.
MERGED_NOTHING = CallResult(
    f"Grouped.\n\n```json\n{json.dumps({'clusters': [[0], [1]]})}\n```\n", 0)
CALL_ERROR = CallResult(
    "", 0, "agent error: error_max_turns: Reached maximum number of turns")
ECHOED = CallResult(f"Here you go.\n\n```json\n{_CLUSTER_TEMPLATE}\n```\n", 0)
NO_GROUPING = CallResult("I could not group these findings.", 0)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, cluster_reply, *, findings_per_persona=True) -> None:
    """Two personas raise one distinct finding each; the clusterer replies as told."""
    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "synthesiser" in mandate:
            return cluster_reply
        if "You are Analyst " in mandate and findings_per_persona:
            return _contract([_finding("first", 1)])
        if "You are Critic " in mandate and findings_per_persona:
            return _contract([_finding("second", 40)])
        return _contract([])

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _reduce_of(repo, capsys, monkeypatch, reply, *extra) -> tuple[dict, dict]:
    _stub(monkeypatch, reply)
    _run(repo, "--semantic-dedup", *extra)
    payload = _artefact(capsys.readouterr().out)
    return payload["reduce"], payload


# --- the defect: a working reduce and a dead one told apart -------------------

@pytest.mark.parametrize("reply, state", [
    (MERGED_NOTHING, "ran"),
    (CALL_ERROR, "call_failed"),
    (ECHOED, "echoed_contract"),
    (NO_GROUPING, "no_grouping"),
])
def test_each_reduce_outcome_is_named(sourced_repo, capsys, monkeypatch, reply, state):
    reduce, _ = _reduce_of(sourced_repo, capsys, monkeypatch, reply)

    assert reduce["requested"] is True
    assert [e["state"] for e in reduce["epochs"]] == [state]


def test_a_dead_reduce_is_distinguishable_from_a_live_one_that_merged_nothing(
        sourced_repo, capsys, monkeypatch):
    """The measurement, as a test: identical clusters, different reported state.

    This is the property the archive's two silently-dead clusterer runs needed and
    did not have. It deliberately asserts that the CLUSTERS are the same, so it
    cannot pass by accident of the two runs differing in some other way.
    """
    live, live_payload = _reduce_of(sourced_repo, capsys, monkeypatch, MERGED_NOTHING)
    dead, dead_payload = _reduce_of(sourced_repo, capsys, monkeypatch, CALL_ERROR)

    assert live_payload["clusters"] == dead_payload["clusters"], \
        "the two runs differ in their clusters, so the test proves nothing"
    assert live["epochs"] != dead["epochs"]
    assert [e["degraded"] for e in live["epochs"]] == [False]
    assert [e["degraded"] for e in dead["epochs"]] == [True]


def test_an_echoed_cluster_contract_is_not_a_failed_call(sourced_repo, capsys,
                                                         monkeypatch):
    """#61 made an echoed cluster contract degrade to identity, correctly — and so
    added a fifth reason to four branches. It must not be reported as a call that
    failed: nothing failed, the model restated our example.
    """
    echoed, _ = _reduce_of(sourced_repo, capsys, monkeypatch, ECHOED)
    failed, _ = _reduce_of(sourced_repo, capsys, monkeypatch, CALL_ERROR)

    assert echoed["epochs"][0]["state"] != failed["epochs"][0]["state"]
    assert echoed["epochs"][0]["degraded"] and failed["epochs"][0]["degraded"]


# --- what is not a degradation ------------------------------------------------

def test_a_run_that_did_not_ask_for_a_reduce_reports_no_degradation(
        sourced_repo, capsys, monkeypatch):
    _stub(monkeypatch, MERGED_NOTHING)
    _run(sourced_repo)                      # no --semantic-dedup
    reduce = _artefact(capsys.readouterr().out)["reduce"]

    assert reduce["requested"] is False
    assert reduce["epochs"] == []
    assert not any(e.get("degraded") for e in reduce["epochs"])


def test_too_few_findings_to_reduce_is_not_a_degradation(sourced_repo, capsys,
                                                         monkeypatch):
    """Fewer than two findings never reaches the clusterer. Nothing degraded."""
    _stub(monkeypatch, MERGED_NOTHING, findings_per_persona=False)
    _run(sourced_repo, "--semantic-dedup")
    reduce = _artefact(capsys.readouterr().out)["reduce"]

    assert [e["state"] for e in reduce["epochs"]] == ["nothing_to_reduce"]
    assert [e["degraded"] for e in reduce["epochs"]] == [False]


def test_a_dry_run_did_not_degrade_its_reduce_either(sourced_repo, capsys, monkeypatch):
    _stub(monkeypatch, MERGED_NOTHING)
    _run(sourced_repo, "--semantic-dedup", "--dry-run")
    reduce = _artefact(capsys.readouterr().out)["reduce"]

    assert [e["state"] for e in reduce["epochs"]] == ["dry_run"]
    assert [e["degraded"] for e in reduce["epochs"]] == [False]


def test_the_operator_is_told_in_prose_not_by_a_ratio(sourced_repo, capsys,
                                                      monkeypatch):
    """The console, not just the artefact: the old tell was a nine-token delta."""
    _stub(monkeypatch, CALL_ERROR)
    _run(sourced_repo, "--semantic-dedup")
    console = capsys.readouterr().out.split("--- JSON ---")[0]

    assert "semantic reduce" in console.lower()
    assert "degraded" in console.lower()


# --- a reply that cannot be read must not cost the run (#149) -------------------

# A million deep, which raises on every interpreter measured (the threshold runs
# from 994 levels on 3.11 to 116,224 on 3.14); 2,000 parsed cleanly on CI's 3.12.
# ``test_reduce_never_raises.py`` asserts the precondition on the interpreter.
NESTED_REPLY = CallResult(
    "```json\n" + "[" * 1_000_000 + "]" * 1_000_000 + "\n```\n", 0)


def test_a_nested_cluster_reply_still_publishes_the_run(sourced_repo, capsys,
                                                        monkeypatch):
    """REGRESSION for #149: on main ``RecursionError`` escaped ``kuang.cli.main``.

    No ``--- JSON ---`` block was printed, so every finding from a panel already paid
    for was lost. The reply is model-authored, so this is a live trigger under
    ``--semantic-dedup``.
    """
    _stub(monkeypatch, NESTED_REPLY)
    try:
        _run(sourced_repo, "--semantic-dedup")
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"escaped main: {type(exc).__name__}") from None
    payload = _artefact(capsys.readouterr().out)

    assert payload["reduce"]["epochs"] == [
        {"epoch": 1, "state": "no_grouping", "degraded": True}]
    assert len(payload["findings"]) == 2


def test_each_epoch_reports_its_own_reduce_state(sourced_repo, capsys, monkeypatch):
    """REGRESSION for #149: the state list is numbered by POSITION, so it must not
    be short.

    Epoch 1's clusterer replies with something unreadable and epoch 2's answers.
    A record left short by epoch 1 would publish epoch 2's ``ran`` as epoch 1's,
    with no epoch 2 at all. Asserted on the whole list, so a shift cannot pass.

    **What kills it today, stated rather than implied.** On main it goes red because
    the run ESCAPES, not because the list shifts: while the engine calls ``reduce``
    bare, a short record and a lost run are the same event. The shift on its own
    becomes reachable only once #148 contains a raising ``reduce``, and that is the
    case this test exists for. The pre-merge review found it was killed by nothing
    else, and the unit test ``test_the_record_has_exactly_one_state_per_call_on_every_path``
    is the one that pins the append directly.
    """
    replies = iter([NESTED_REPLY, MERGED_NOTHING])
    _stub(monkeypatch, None)
    real = session_module.PanelSession._run_claude

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001
        if "synthesiser" in mandate:
            return next(replies)
        return real(self, prompt, mandate, tools, model)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)
    try:
        _run(sourced_repo, "--semantic-dedup", "--max-epochs", "2",
             "--stall-patience", "2")
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"escaped main: {type(exc).__name__}") from None
    payload = _artefact(capsys.readouterr().out)

    assert payload["reduce"]["epochs"] == [
        {"epoch": 1, "state": "no_grouping", "degraded": True},
        {"epoch": 2, "state": "ran", "degraded": False}]


def test_a_clusterer_call_that_raises_still_publishes_the_run(sourced_repo, capsys,
                                                              monkeypatch):
    """REGRESSION for #149: on main the raise escaped ``kuang.cli.main``.

    Added at the pre-merge review, which found the CALL guard was killed only by
    unit tests: removing it turned nothing red end to end.
    """
    _stub(monkeypatch, None)
    real = session_module.PanelSession._run_claude

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001
        if "synthesiser" in mandate:
            raise FileNotFoundError("no claude binary here")
        return real(self, prompt, mandate, tools, model)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)
    try:
        _run(sourced_repo, "--semantic-dedup")
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"escaped main: {type(exc).__name__}") from None
    payload = _artefact(capsys.readouterr().out)

    assert payload["reduce"]["epochs"] == [
        {"epoch": 1, "state": "call_failed", "degraded": True}]
    assert len(payload["findings"]) == 2
