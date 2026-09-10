# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The panel is asked to show its working, and the run must not throw it away (#112).

``Finding.evidence`` had seven writers in ``kuang/`` and no reader at all: the
output contract asks each persona for ``"evidence": "what you checked and found"``
(``contract.py:30``), every finding carried the answer, and no run ever emitted
one. A run read back cold recorded *that* a persona claimed something, never
*what it checked* — and since #111 six of those seven writers mark a truncation
that no channel could display, so that fix was unobservable by construction.

The channel is the **artefact**, and only the artefact. Two rejections are as
load-bearing as the emission and are pinned here rather than left to drift:

* not ``report.ledger_line`` — cross-epoch memory and the ``--prior-findings``
  seed share that renderer, so evidence there would put a median ~355 characters
  of model prose about the surface into the next epoch's prompt and into a later
  run's seed (#71's contamination, #63's trust boundary);
* not the console — which prints only open uglies and blockers, so a console
  channel would publish the *review* prose while still not showing the parser's
  own diagnoses, which are NOTEs. The geometry is upside-down; the artefact is
  the only channel carrying both populations.

The bound rides at the publish site, which is safe here and was not at #33:
``evidence`` is no part of ``Finding.key``, so a cap moves no dedup signature and
therefore no stall counter. ``evidence_chars`` is ``surface_lost.detail_chars``'
shape (#111) — the structural half, so "was this cut" is exactly knowable without
reading the marker's wording back.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — nothing spawned, no network.

Deliberately **not** asserted here: that an operator reading only the
human-readable half of stdout can read a persona's diagnosis. They cannot, that
is the stated under-report of the channel decision, and the run's own account of
itself is #117. Nor does anything here touch #63: this publishes a field whose
content #63 is about, and fixes nothing about the retry that can originate it.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.memory import epoch_summary
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import EVIDENCE_BOUND, main
from kuang.engine import Finding, Severity
from kuang.report import ledger_line

# Three named experts, and no fourth: "completeness" and "ruin" appear in the
# bodies, so ``_ensure_specialists`` injects neither default.
_CLAUDE_MD = """\
# A repo

# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?

## Critic — Completeness

Find what everyone else missed; assume shared blind spots.

## Sentinel — Ruin

Hunt ruin-class hazards only.
"""

# The longest ``evidence`` in the 19-envelope probe corpus, n=51 findings (47 from
# personas, 4 parser diagnoses; 44 of the 51 from the seven live captures),
# measured 2026-09-10 by parsing each envelope through ``parse_findings``. The
# bound is derived from this number and the test below holds it to that, so the
# constant cannot quietly drift down into the distribution it is meant to sit
# above. Median was 346 and 19 of the 51 exceeded ``DIAGNOSIS_BOUND``'s 400.
_MEASURED_MAX_EVIDENCE = 1038

# A string no title, persona name or path can collide with, so an assertion that
# the console does NOT carry it cannot be satisfied by accident (#87, #103).
_SENTINEL = "ZQX-evidence-sentinel-7413"


def _contract(verdict="YES", findings=()) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


def _finding(*, title="a finding", evidence="read a.py and traced the caller",
             claim_class="correctness", severity="BLOCKER", line=10):
    """One ordinary finding. ``evidence=None`` omits the key, as a persona may."""
    f = {"title": title, "severity": severity, "claim_class": claim_class,
         "file": "a.py", "line": line}
    if evidence is not None:
        f["evidence"] = evidence
    return f


_CLEAN = _contract()
_NO_JSON = CallResult("I read the diff and have nothing structured to say.", 0)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=_CLEAN, formatter=_NO_JSON) -> None:
    """Reply per persona per epoch: ``{name: [epoch1, epoch2, ...]}``."""
    seen: dict[str, int] = {}

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate:
            return _NO_JSON
        if "formatter" in mandate:
            return formatter
        for name, per_epoch in replies.items():
            if f"You are {name} " in mandate:
                i = seen.get(name, 0)
                seen[name] = i + 1
                return per_epoch[min(i, len(per_epoch) - 1)]
        return default

    monkeypatch.setattr(session_module.PanelSession, "_run_claude", _fake)


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _console(out: str) -> str:
    """The human-readable half only: everything before the artefact block."""
    return out.split("--- JSON ---")[0]


def _by_title(findings: list[dict], title: str) -> dict:
    """The one record with this title.

    Every assertion below is anchored to the record it is about rather than
    searched for across the output: ``evidence`` is model prose that may contain
    the title, a persona name or a path, so a substring assertion over the whole
    console or artefact is guaranteed to be killed by nothing (#87, #103).
    """
    matches = [f for f in findings if f["title"] == title]
    assert len(matches) == 1, f"expected one {title!r}, got {len(matches)}"
    return matches[0]


# --- 1. the reviews: what a persona says it checked ---------------------------

def test_the_artefact_carries_what_the_persona_says_it_checked(
        sourced_repo, monkeypatch, capsys):
    """The ordinary path, and the one the issue is titled for."""
    working = f"read a.py:1-20, ran the branch by hand, {_SENTINEL}"
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding(evidence=working)])]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    assert findings, "the fixture raised nothing to publish"
    assert _by_title(findings, "a finding")["evidence"] == working


def test_the_artefact_carries_the_parsers_own_diagnosis(
        sourced_repo, monkeypatch, capsys):
    """The other population, and the reason #111 could not be demonstrated.

    A reply whose ``findings`` is an object takes the contract-miss path, whose
    finding keeps the payload that says *why* the reply could not be read. Six of
    #111's seven marked truncations are on this field; a marker on a field no
    channel emits is a fix nobody can watch work.
    """
    payload = {"verdict": "NO", "findings": {"not": "a list"}}
    reply = CallResult(f"```json\n{json.dumps(payload)}\n```\n", 0)
    _stub(monkeypatch, {"Analyst": [reply]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    diagnosis = _by_title(
        findings, "findings contract violated: 'findings' was a dict, not a list")
    assert diagnosis["about_run"] is True
    # Compared as JSON, not as text: what must survive is the payload an operator
    # needs to see, not the parser's incidental whitespace.
    assert json.loads(diagnosis["evidence"]) == payload


# --- 2. the structural half: the loss is exactly knowable ---------------------

def test_every_finding_carries_the_length_its_evidence_had(
        sourced_repo, monkeypatch, capsys):
    """Always present, never optional — an absent value makes no claim (#111).

    Including the finding whose persona wrote nothing: the measured minimum is
    zero characters, so without this a reader could not tell "the persona said
    nothing" from "the run declined to publish it".
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [
        _finding(title="said what it checked", evidence="read a.py", claim_class="c1"),
        _finding(title="said nothing", evidence=None, claim_class="c2")])]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    assert all("evidence_chars" in f for f in findings), \
        "a finding record omits the length its evidence had before the bound"
    assert _by_title(findings, "said what it checked")["evidence_chars"] == len("read a.py")
    silent = _by_title(findings, "said nothing")
    assert silent["evidence"] == ""
    assert silent["evidence_chars"] == 0


def test_over_long_evidence_is_cut_marked_and_its_true_length_reported(
        sourced_repo, monkeypatch, capsys):
    """One persona reply may not put an unbounded string into a shared artefact.

    The tail is a sentinel rather than more filler: a homogeneous string cannot
    express this axis, because its tail is a substring of its own head and the
    "the tail was cut" assertion below passes whatever the code does.
    """
    raw = "A" * EVIDENCE_BOUND + "-tail-" + _SENTINEL
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding(evidence=raw)])]})
    _run(sourced_repo)
    record = _by_title(_artefact(capsys.readouterr().out)["findings"], "a finding")

    assert record["evidence_chars"] == len(raw), \
        "the record must say how long the evidence was BEFORE the bound"
    assert record["evidence"].startswith(raw[:EVIDENCE_BOUND])
    assert raw[EVIDENCE_BOUND:] not in record["evidence"], "the tail was not cut"
    # The marker rides on TOP of the bound, as it does at ``surface.py:219``:
    # paying for it out of the bound would make the limit mean two things.
    assert len(record["evidence"]) > EVIDENCE_BOUND


def test_evidence_within_the_bound_is_published_whole_and_unmarked(
        sourced_repo, monkeypatch, capsys):
    """The mirror image, and the probe a "mark everything" implementation fails.

    A rule that fires on a healthy run is the mirror image of the defect it is
    meant to prevent, so this forbids the shape that would make the test above
    pass trivially.
    """
    raw = "read a.py and the two callers; " + "traced it. " * 20
    assert len(raw) < EVIDENCE_BOUND, "the fixture cannot express the axis"
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding(evidence=raw)])]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    assert _by_title(findings, "a finding")["evidence"] == raw
    assert all(len(f["evidence"]) == f["evidence_chars"] for f in findings), \
        "a run that cut nothing must not report a cut anywhere"


def test_the_bound_sits_above_the_measured_distribution(sourced_repo):
    """The bound is derived from a measurement, not borrowed from a neighbour.

    GREEN ON MAIN, and labelled rather than left to look like a regression test:
    nothing published ``evidence`` before this change, so there was no bound to be
    wrong. This is a guard against the option that was REJECTED — reusing
    ``DIAGNOSIS_BOUND`` — and it goes red the moment somebody adopts it.

    A published bound on model prose sits ABOVE the measured distribution of the
    field it bounds, so it fires on an outlier and never on an ordinary review.
    ``DIAGNOSIS_BOUND``'s 400 fails that test here by measurement rather than by
    doctrine: 19 of the 51 corpus findings exceed it, all 19 of them from live
    captures — 43% of a real run truncated.
    """
    assert EVIDENCE_BOUND > _MEASURED_MAX_EVIDENCE


# --- 3. the channels this deliberately does NOT reach -------------------------

def test_the_cluster_members_do_not_repeat_the_evidence(
        sourced_repo, monkeypatch, capsys):
    """``members[]`` is a thinner projection of a finding published in full.

    GREEN ON MAIN: members never carried ``evidence``, so this is a guard against
    a widening rather than a regression test. It is here because the question
    arrives with the key and is better answered than noticed later.

    It already omits ``about_run``, ``ungrounded`` and ``claim_class``;
    ``evidence`` follows the same rule, so the question is answered here rather
    than noticed later.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding()])]})
    _run(sourced_repo)
    clusters = _artefact(capsys.readouterr().out)["clusters"]

    assert clusters, "the fixture produced no clusters to check"
    assert all("evidence" not in m and "evidence_chars" not in m
               for c in clusters for m in c["members"])


def test_the_console_does_not_carry_the_evidence(
        sourced_repo, monkeypatch, capsys):
    """The rejected channel, pinned so widening it is a decision, not a drift.

    Red on main only through its first assertion, which is the guard that stops
    the second from proving nothing: before this change the artefact carried no
    ``evidence`` either, so an absence in the console said nothing at all.

    The console prints only open uglies and blockers, so it would carry the
    persona's prose — the untrusted half — and still not the parser's diagnoses.
    Asserted with a sentinel no other field can collide with, so the absence is
    real rather than an artefact of the search.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [
        _finding(evidence=f"checked it thoroughly {_SENTINEL}")])]})
    _run(sourced_repo)
    out = capsys.readouterr().out

    assert _SENTINEL in _by_title(_artefact(out)["findings"], "a finding")["evidence"], \
        "the fixture did not reach the artefact, so the absence below proves nothing"
    assert _SENTINEL not in _console(out)


def test_cross_epoch_memory_does_not_carry_the_evidence():
    """The other rejected channel, and the one with a compounding cost (#71).

    GREEN ON MAIN, and deliberately so: this pins a channel decision, not a fix.
    It is the rejected option expressed as a test, and it goes red on any change
    that routes ``evidence`` through the shared ledger renderer.

    ``epoch_summary`` and ``load_prior_findings`` share one renderer, so evidence
    in a ledger line would reach the next epoch's prompt AND a later run's seed —
    model prose about a surface the next panel never saw. Asserted as an equality
    against the same finding rendered with no evidence at all, so a widening
    cannot hide inside a substring.
    """
    f = Finding("Analyst", "a finding", Severity.BLOCKER, "correctness", "a.py", 10,
                evidence=f"read it {_SENTINEL}")

    assert epoch_summary([f]) == ledger_line(
        severity="BLOCKER", is_open=True, persona="Analyst", title="a finding",
        file="a.py", line=10)
