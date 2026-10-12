# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run whose injected callables failed must publish that, and still publish itself (#148).

The engine half is ``tests/engine/test_unguarded_seams.py``: each callable is guarded,
UNKNOWN is substituted, and a ``SeamFailure`` is recorded on the epoch. This is the
published half: one new top-level key, ``seam_failures``, always present and ``[]``
where nothing failed, in ``gate_failures``' shape with ``seam`` naming the callable;
a console exception section; and the exit code for ``unmeasured``.

The shipped CLI injects ``checkpoint`` (``session.on_epoch``) and ``budget_spent``
(``session.budget_spent``), so those two are the seams driven here, stubbed at the
session, which is where the engine calls them. Nothing is spawned; no network.

**Which are regressions, measured on main with this file in place.** Recorded in the
PR body; any test green on main says so in its own docstring.
"""

from __future__ import annotations

import json

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
"""


def _contract(epoch: int) -> CallResult:
    """One open BLOCKER, distinct per epoch so a second epoch is reached at all."""
    body = json.dumps({"verdict": "NO", "findings": [
        {"title": f"off-by-one in the retry bound (epoch {epoch})",
         "severity": "BLOCKER", "claim_class": f"correctness-{epoch}",
         "file": "a.py", "line": 1}]})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```", 18, num_turns=7)


class _SeamError(RuntimeError):
    """Stands in for what a durable write would raise once #31 lands."""


def _stub(monkeypatch, changed_repo, **seams) -> None:
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    epochs = {"n": 0}
    real_gather = session_module.PanelSession.gather

    def _fake_call(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate or "formatter" in mandate:
            return CallResult("nothing structured to say.", 0)
        return _contract(epochs["n"])

    def _gather(self, epoch: int) -> str:  # noqa: ANN001
        epochs["n"] = epoch
        return real_gather(self, epoch)

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake_call)
    monkeypatch.setattr(session_module.PanelSession, "gather", _gather)
    for name, seam in seams.items():
        monkeypatch.setattr(session_module.PanelSession, name, seam)


def _raising(message="disk full"):
    def seam(self, *a, **k):  # noqa: ANN001, ARG001
        raise _SeamError(message)
    return seam


def _run(repo, *extra) -> int:
    try:
        return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "2",
                     "--stall-patience", "2", "--no-parallel", *extra])
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"escaped main: {type(exc).__name__}") from None


def _split(out: str) -> tuple[str, dict]:
    console, _, artefact = out.rpartition("--- JSON ---")
    return console, json.loads(artefact)


def test_a_failing_checkpoint_is_published_and_the_run_survives(changed_repo, capsys,
                                                               monkeypatch):
    """REGRESSION for #148: on main the raise escaped ``main`` and nothing was printed.

    #31 is what makes this live, by making ``checkpoint`` a durable write.
    """
    _stub(monkeypatch, changed_repo, on_epoch=_raising())
    rc = _run(changed_repo)
    console, payload = _split(capsys.readouterr().out)

    assert rc == 0
    assert payload["halt_reason"] == "epoch"
    assert payload["seam_failures"] == [
        {"epoch": 1, "seam": "checkpoint", "detail": "_SeamError: disk full",
         "detail_chars": 21},
        {"epoch": 2, "seam": "checkpoint", "detail": "_SeamError: disk full",
         "detail_chars": 21}]
    assert [ln for ln in console.splitlines() if ln.startswith("seam failed:")] == [
        "seam failed: epoch 1, checkpoint — _SeamError: disk full",
        "seam failed: epoch 2, checkpoint — _SeamError: disk full"]


def test_an_unreadable_token_counter_with_a_ceiling_halts_unmeasured(changed_repo,
                                                                    capsys,
                                                                    monkeypatch):
    """REGRESSION for #148, the ruling end to end: halt ``unmeasured``, exit 2.

    Not ``budget`` (nothing measured a ceiling as reached), and not 0, which a caller's
    ``&&`` would read as success — ``surface_lost``'s argument, which #32 owns.
    """
    _stub(monkeypatch, changed_repo, budget_spent=_raising("counter unavailable"))
    rc = _run(changed_repo, "--token-budget", "100000")
    console, payload = _split(capsys.readouterr().out)

    assert rc == 2
    assert payload["halt_reason"] == "unmeasured"
    assert [r["seam"] for r in payload["seam_failures"]] == ["budget_spent"]
    assert "=== HALT: unmeasured after 1 epoch(s) ===" in console
    assert "seam failed: epoch 1, budget_spent — _SeamError: counter unavailable" in console


def test_an_unreadable_token_counter_with_no_ceiling_is_recorded_and_continues(
        changed_repo, capsys, monkeypatch):
    """REGRESSION for #148: the value could not change the halt, and killed the run."""
    _stub(monkeypatch, changed_repo, budget_spent=_raising("counter unavailable"))
    rc = _run(changed_repo)
    _, payload = _split(capsys.readouterr().out)

    assert rc == 0
    assert payload["halt_reason"] == "epoch"
    assert [(r["epoch"], r["seam"]) for r in payload["seam_failures"]] == [
        (1, "budget_spent"), (2, "budget_spent")]


def test_the_published_diagnosis_is_bounded_and_says_how_long_it_was(changed_repo,
                                                                    capsys,
                                                                    monkeypatch):
    """REGRESSION for #148: the pair #111 requires, as ``gate_failures`` publishes it."""
    _stub(monkeypatch, changed_repo, on_epoch=_raising("x" * 1000))
    _run(changed_repo, "--max-epochs", "1")
    _, payload = _split(capsys.readouterr().out)
    row = payload["seam_failures"][0]

    assert row["detail_chars"] == len("_SeamError: ") + 1000
    assert len(row["detail"]) < row["detail_chars"]


def test_a_healthy_run_publishes_an_empty_list_and_prints_no_section(changed_repo,
                                                                    capsys,
                                                                    monkeypatch):
    """The pair: present and ``[]``, and no console line, on a run where nothing failed.

    Red on main only on the missing key. An absent key makes no claim, and a section
    printed on a healthy run would be the rule firing where it must not.
    """
    _stub(monkeypatch, changed_repo)
    rc = _run(changed_repo)
    console, payload = _split(capsys.readouterr().out)

    assert rc == 0
    assert payload["seam_failures"] == []
    assert "seam failed:" not in console
