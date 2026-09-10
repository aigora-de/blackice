# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The ledger line is a contract between two modules, and nothing enforced it.

``on_epoch`` renders ``- [SEV/state] (persona) title @ loc`` into this run's
cross-epoch memory; the CLI writes the same findings out as JSON;
``load_prior_findings`` reads that JSON back on a later run (``--prior-findings``)
and renders the line *again*, from a second implementation.

So the two renderers must agree byte-for-byte, and until #19 gave them one
implementation nothing checked that they did. These are regression tests over the
whole round trip — ``on_epoch`` → the CLI's JSON → ``load_prior_findings`` —
rather than over either renderer alone, because agreeing is the property that
matters and either half can drift on its own.

The line carries per-finding *provenance* as well as state (#71): whether a
finding is the instrument's own diagnosis (``about_run``, #73) and whether the
call that produced it opened nothing (``ungrounded``, #70). Both cross the round
trip, which is what makes a contaminated artefact seed a later run contaminated
rather than clean (#13).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kuang.backends.claude_code import PanelSession, load_prior_findings
from kuang.cli import EVIDENCE_BOUND
from kuang.engine import (Finding, PersonaReport, PersonaStatus, ReviewSpec,
                          Severity, bounded_diagnosis)


def _cli_json(findings, ungrounded=()):
    """The findings array exactly as ``kuang.cli.main`` writes it under '--- JSON ---'.

    Kept faithful deliberately. It carried neither ``about_run`` (added by #73) nor
    ``ungrounded`` (#71) while claiming to mirror the record, which is the shape
    #87 was filed for — a fixture that claims a fidelity it does not have. The
    round trip cannot prove agreement about a key the fixture omits, and it cannot
    prove a key is IGNORED unless the fixture carries it either — which is what
    ``claim_class`` (#33) and ``evidence`` (#112) are doing here. The bound rides
    on the mirror too: the CLI publishes the value it bounds, so a fixture
    publishing the raw string would mirror a record the CLI never writes.
    """
    return {"findings": [
        {"persona": f.persona, "severity": f.severity.name, "title": f.title,
         "file": f.file, "line": f.line, "open": f.counts_open,
         "about_run": f.about_run, "ungrounded": f.key in set(ungrounded),
         "claim_class": f.claim_class,
         "evidence": bounded_diagnosis(f.evidence, limit=EVIDENCE_BOUND),
         "evidence_chars": len(f.evidence)}
        for f in findings]}


def _round_trip(tmp_path, findings, ungrounded=()) -> tuple[str, str]:
    """Render the same findings both ways: (this run's memory, a later run's seed).

    Drives the real ``on_epoch``, which is the left-hand end of the round trip the
    module docstring names — including the (persona, epoch) join it performs to
    decide which lines are ``[ungrounded]``.
    """
    session = PanelSession(repo_root=tmp_path, spec=ReviewSpec(why="w", what="x"),
                           personas={}, base="")
    marked = set(ungrounded)
    reports = [
        PersonaReport(persona="P1", status=PersonaStatus.CONTRIBUTED, turns=turns,
                      findings=[f for f in findings if (f.key in marked) is degraded])
        for turns, degraded in ((1, True), (7, False))]
    session.on_epoch(SimpleNamespace(
        ledger={f.key: f for f in findings},
        epochs=[SimpleNamespace(index=1, reports=reports)]))

    path = tmp_path / "findings.json"
    path.write_text(json.dumps(_cli_json(findings, ungrounded)))

    return session.prior_summary, load_prior_findings(path)


@pytest.mark.parametrize("finding", [
    Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120),
    Finding("P2", "stale threshold", Severity.NOTE, "calc", "calc.py", 7, verified=False),
    Finding("P3", "ruin-class data loss", Severity.UGLY, "drop", "a.py", 1),
    Finding("P4", "no location at all", Severity.NON_BLOCKING, "meta"),
    Finding("P5", "a file but no line", Severity.NOTE, "meta", "only.py"),
])
def test_the_two_renderers_agree(tmp_path, finding):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding])

    assert epoch_memory == seeded_memory


def test_they_agree_over_a_whole_ledger(tmp_path):
    findings = [
        Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120),
        Finding("P2", "stale threshold", Severity.NOTE, "calc", "calc.py", 7, verified=False),
        Finding("P3", "no location", Severity.UGLY, "meta"),
    ]

    epoch_memory, seeded_memory = _round_trip(tmp_path, findings)

    assert epoch_memory == seeded_memory
    assert len(epoch_memory.splitlines()) == 3


def test_the_line_still_looks_like_this(tmp_path):
    """One literal assertion, so the shared format cannot drift silently."""
    epoch_memory, seeded_memory = _round_trip(tmp_path, [
        Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry", "runner.py", 120)])

    assert epoch_memory == "- [BLOCKER/open] (P1) unbounded retry loop @ runner.py:120"
    assert seeded_memory == epoch_memory


def test_a_resolved_finding_reads_as_resolved_on_both_sides(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [
        Finding("P2", "fixed since", Severity.BLOCKER, "x", "a.py", 3, verified=False)])

    assert epoch_memory == "- [BLOCKER/resolved] (P2) fixed since @ a.py:3"
    assert seeded_memory == epoch_memory


def test_the_dedup_category_reaches_the_artefact_and_not_the_line(tmp_path):
    """#33 publishes ``claim_class``; the seed line is deliberately unchanged.

    The two ends of the round trip are asked different questions, and this is where
    the difference is asserted rather than assumed. The ARTEFACT must carry the
    category, because the dedup signature has to be reconstructable from an
    archived run. The LINE must not, because the seed is **prose fed to a model**,
    not data: a seeded finding never enters ``review_run.ledger``, so a tag here
    could not make dedup carry across runs — it would only look as though it did,
    while spending prompt budget and moving a byte-for-byte contract.

    The tag slot is also not free for it. ``ledger_line``'s tags are facts OUR CODE
    sets — ``[about the run]`` (#73), ``[ungrounded]`` (#70) — and #73 measured what
    putting a model-authored value where our own flags live costs. #63's marker for
    a retry-originated finding is the natural next occupant.

    MUTATION: render ``claim_class`` into ``ledger_line`` -> this and
    ``test_the_line_still_looks_like_this`` both go red.

    **This is a guard, not a regression test, and it passes on ``main``** — measured,
    not assumed. Main already has the property it names (the line ignores the
    category), and the artefact half is satisfied by ``_cli_json``, which is this
    module's own mirror of the CLI array rather than the CLI. It goes red under the
    REJECTED option, never under the defect; the artefact key itself is held by
    ``cli/test_halt_input_reporting.py``. Recorded here because a test that dies to
    nothing is either a gap or needs justifying, and this one needs justifying.
    """
    # A category that appears nowhere in the rendered line, so "did it leak?" is a
    # question the fixture can actually answer: the obvious ``"retry"`` is a
    # substring of the title above it, and searching for it would be killed by
    # nothing (#87, and #103's console assertion one layer along).
    finding = Finding("P1", "unbounded retry loop", Severity.BLOCKER,
                      "resource-exhaustion", "runner.py", 120)
    payload = _cli_json([finding])
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding])

    assert payload["findings"][0]["claim_class"] == "resource-exhaustion", \
        "the artefact cannot rebuild the dedup signature without the category"
    assert epoch_memory == "- [BLOCKER/open] (P1) unbounded retry loop @ runner.py:120", \
        "the category leaked into the line the next panel reads"
    assert epoch_memory == seeded_memory, \
        "a key the artefact carries and the line ignores broke the round trip"


def test_the_evidence_reaches_the_artefact_and_not_the_line(tmp_path):
    """#112 publishes ``evidence``; the seed line is deliberately unchanged.

    The same question ``claim_class`` answered above, on a field with a
    compounding cost rather than a merely useless one. The ARTEFACT carries the
    persona's "what I checked", because a run read back cold has to record the
    check and not only the claim. The LINE must not: ``epoch_summary`` and
    ``load_prior_findings`` share this renderer, so a median ~355 characters of
    model prose about the surface would enter the next epoch's prompt AND a later
    run's seed — one failed call's fabrication handed to personas that did not
    fail (#71), through the trust boundary #63 owns.

    MUTATION: render ``evidence`` into ``ledger_line`` -> this and
    ``test_the_line_still_looks_like_this`` both go red.

    **A guard, not a regression test, and it passes on ``main``** — measured, like
    its sibling above. Main has the line half already; the artefact half is held by
    ``cli/test_evidence_reporting.py``, and here by ``_cli_json``'s mirror. What is
    load-bearing is that the two ends stay ASYMMETRIC and the round trip survives
    it.
    """
    # Prose that shares no substring with the rendered line, so "did it leak?" is a
    # question the fixture can answer (#87).
    finding = Finding("P1", "unbounded retry loop", Severity.BLOCKER,
                      "resource-exhaustion", "runner.py", 120,
                      evidence="opened the queue module and counted the wakeups")
    payload = _cli_json([finding])
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding])

    assert payload["findings"][0]["evidence"] == \
        "opened the queue module and counted the wakeups", \
        "the artefact records the claim and not the check"
    assert payload["findings"][0]["evidence_chars"] == len(finding.evidence)
    assert epoch_memory == "- [BLOCKER/open] (P1) unbounded retry loop @ runner.py:120", \
        "model prose about the surface leaked into the line the next panel reads"
    assert epoch_memory == seeded_memory, \
        "a key the artefact carries and the line ignores broke the round trip"


def test_a_seed_from_an_artefact_carrying_evidence_reads_identically(tmp_path):
    """Published, never read back: the seed must ignore the key, not consume it.

    ``load_prior_findings`` reads eight keys and ``evidence`` is not one of them.
    That is the boundary #73 measured the cost of crossing — our own code reading
    model-authored data — and it is asserted as an equality between a seed taken
    from an artefact WITH the key and one taken from an artefact without it, so a
    later reader that starts consuming it cannot do so silently.
    """
    finding = Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry",
                      "runner.py", 120, evidence="read every caller of enqueue()")
    with_key = _cli_json([finding])
    without_key = {"findings": [
        {k: v for k, v in f.items() if not k.startswith("evidence")}
        for f in _cli_json([finding])["findings"]]}
    assert "evidence" in with_key["findings"][0], "the fixture cannot express the axis"

    paths = []
    for name, payload in (("with.json", with_key), ("without.json", without_key)):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        paths.append(path)

    assert load_prior_findings(paths[0]) == load_prior_findings(paths[1])


def test_a_seed_whose_finding_has_no_line_key_still_loads(tmp_path):
    """REGRESSION for #25: ``f["line"]`` raised KeyError on that artefact.

    An artefact written by anything other than this CLI — or by a future version
    that omits an absent field — used to take the seeded re-run down at startup,
    which is the one moment the operator has the least context to interpret it.
    """
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"findings": [
        {"persona": "P1", "severity": "BLOCKER", "title": "t", "file": "a.py",
         "open": True}]}))

    assert load_prior_findings(path) == "- [BLOCKER/open] (P1) t @ a.py:None"


def test_there_is_exactly_one_renderer():
    """The property the round-trip depends on, stated directly."""
    from kuang import report
    from kuang.backends.claude_code import memory

    assert memory.ledger_line is report.ledger_line


# --- provenance crosses the round trip too (#71) ------------------------------

_MARKED = Finding("P1", "unbounded retry loop", Severity.BLOCKER, "retry",
                  "runner.py", 120)
_DIAGNOSIS = Finding("P1", "agent error: boom", Severity.NOTE, "meta",
                     about_run=True)


@pytest.mark.parametrize("finding, ungrounded", [
    (_MARKED, ()),
    (_MARKED, (_MARKED.key,)),
    (_DIAGNOSIS, ()),
    (_DIAGNOSIS, (_DIAGNOSIS.key,)),
], ids=["clean", "ungrounded", "about-the-run", "both"])
def test_the_two_renderers_agree_about_provenance(tmp_path, finding, ungrounded):
    """Every combination, because either half can drift on its own."""
    epoch_memory, seeded_memory = _round_trip(tmp_path, [finding], ungrounded)

    assert epoch_memory == seeded_memory


def test_an_ungrounded_line_still_looks_like_this(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [_MARKED], (_MARKED.key,))

    assert epoch_memory == ("- [BLOCKER/open] (P1) unbounded retry loop "
                            "@ runner.py:120 [ungrounded]")
    assert seeded_memory == epoch_memory


def test_a_line_about_the_run_still_looks_like_this(tmp_path):
    epoch_memory, seeded_memory = _round_trip(tmp_path, [_DIAGNOSIS])

    assert epoch_memory == "- [NOTE/open] (P1) agent error: boom @ - [about the run]"
    assert seeded_memory == epoch_memory


def test_an_artefact_that_predates_the_marks_seeds_exactly_as_before(tmp_path):
    """A saved run from before #71 carries neither key; it must render unchanged."""
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"findings": [
        {"persona": "P1", "severity": "BLOCKER", "title": "t", "file": "a.py",
         "line": 3, "open": True}]}))

    assert load_prior_findings(path) == "- [BLOCKER/open] (P1) t @ a.py:3"
