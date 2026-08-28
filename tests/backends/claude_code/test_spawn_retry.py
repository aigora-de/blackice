# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The reformat retry: a reply that missed the contract is recovered, not lost.

``session.spawn`` reformats a persona's raw review into the contract via one cheap
follow-up call when ``_is_parse_failure`` recognises a contract miss. That path
had no test, and #25 makes a second class of reply eligible for it — a payload
whose ``findings`` is not a list — so the path is now load-bearing for two
failure modes rather than one.

Hermetic: ``_run_claude`` is stubbed, so nothing spawns and no network is touched.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import PanelSession
from kuang.backends.claude_code.personas import Persona
from kuang.backends.claude_code.spawn import CallResult
from kuang.engine import ReviewSpec


def _reply(payload: str) -> str:
    return f"Here is my review.\n\n```json\n{payload}\n```\n"


_RECOVERED = _reply(json.dumps({"verdict": "NO", "findings": [
    {"title": "recovered finding", "severity": "BLOCKER", "claim_class": "x"}]}))


@pytest.fixture
def session(tmp_path):
    return PanelSession(repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
                        personas={"p": Persona("p", "a lens")}, base="")


def _stub(monkeypatch, session, replies):
    """Return the given replies in order, recording the prompts they answered."""
    prompts: list[str] = []

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        prompts.append(prompt)
        return CallResult(replies[len(prompts) - 1], 7, None)

    monkeypatch.setattr(PanelSession, "_run_claude", _fake)
    return prompts


@pytest.mark.parametrize("miss", [
    "I reviewed it and found nothing worth a block.",     # no fenced block at all
    _reply('{"verdict": "NO", "findings": "none"}'),      # #25: findings not a list
])
def test_a_contract_miss_is_reformatted_rather_than_discarded(session, monkeypatch, miss):
    prompts = _stub(monkeypatch, session, [miss, _RECOVERED])

    report = session.spawn("p", "mandate", "surface", 1)

    assert [f.title for f in report.findings] == ["recovered finding"]
    assert len(prompts) == 2, "the retry did not fire"
    assert "EXACT JSON" in prompts[1]
    assert report.tokens == 14, "both calls must be charged to the persona"


def test_a_good_reply_is_not_retried(session, monkeypatch):
    prompts = _stub(monkeypatch, session, [_RECOVERED])

    report = session.spawn("p", "mandate", "surface", 1)

    assert len(prompts) == 1
    assert report.tokens == 7
    assert [f.title for f in report.findings] == ["recovered finding"]
