# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must record the inputs to its own halt (#33, folded #114).

Three values the control logic decides on never reached the artefact: the
``claim_class`` that is the fourth component of the dedup signature, each
persona's verdict (quorum is a conjunct of CONVERGED), and the material
new-cluster count the stall counter is taken on.

The doctrine is #24's and #26's, one step further along. Those two report a value
the run **refused to read**; this reports the values it **did** read and acted on,
so an archived run can be asked why two findings merged and who voted YES.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — persona sourcing, prompt assembly,
``parse_findings``, the engine loop, the printed summary and the ``--- JSON ---``
artefact — with nothing spawned and no network.

Console assertions are anchored to the **line** they are about, never searched for
across the whole output: at #103 a mutation dropping names from a new warning was
killed by nothing, because a line two above carried the same text (#87).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import main

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


def _contract(verdict="YES", findings=()) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


def _finding(claim_class, *, line=10, severity="BLOCKER", title="a finding"):
    return {"title": title, "severity": severity, "claim_class": claim_class,
            "file": "a.py", "line": line, "evidence": "read it"}


_CLEAN = _contract()
_NO_JSON = CallResult("I read the diff and have nothing structured to say.", 0)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=_CLEAN, formatter=_NO_JSON,
          cluster=None) -> None:
    """Reply per persona per epoch: ``{name: [epoch1, epoch2, ...]}``.

    The mandate carries the persona's name; a per-persona call counter carries
    the epoch, since the loop calls the same seam again each epoch.
    """
    seen: dict[str, int] = {}

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        if "careful synthesiser" in mandate:
            return cluster if cluster is not None else _NO_JSON
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


def _signature(record: dict) -> str:
    """Recompute ``Finding.key`` from the published record alone.

    This is the point of publishing ``claim_class``: the dedup signature must be
    reconstructable from an archived run, with no access to the objects that
    produced it. Kept as a literal restatement of ``Finding.key`` rather than a
    call into it, because what is under test is whether the ARTEFACT carries
    enough to rebuild the signature — importing the property would test the
    property instead.
    """
    line = record["line"]
    bucket = "" if line is None else str(line // 10)
    raw = f"{record['file']}|{bucket}|{record['claim_class']}|{record['severity']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _synthesis_lines(out: str) -> list[str]:
    """The gate's count line for each epoch that printed one, in order.

    Anchored: the line AFTER each ``=== epoch N synthesis ===`` header, never a
    search across the whole console (#87).
    """
    lines = out.splitlines()
    return [lines[i + 1] for i, line in enumerate(lines)
            if line.startswith("=== epoch ") and line.endswith(" synthesis ===")]


# --- 1. the category the dedup signature is taken on -------------------------

def test_the_artefact_carries_the_category_the_signature_is_taken_on(
        sourced_repo, monkeypatch, capsys):
    """``claim_class`` decides the signature and never reached the artefact."""
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding("correctness")])]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    assert findings, "the fixture raised nothing to publish"
    assert all("claim_class" in f for f in findings), \
        "a finding record omits the field its dedup signature is taken on"
    assert {f["claim_class"] for f in findings} == {"correctness"}


def test_the_dedup_signature_is_reconstructable_from_the_artefact_alone(
        sourced_repo, monkeypatch, capsys):
    """Two findings differing ONLY by category must not read as one.

    The load-bearing case, and the reason a presence assertion is not enough: the
    two records below agree on file, line-bucket and severity, so ``claim_class``
    is the whole of what separates their signatures. A field published as a
    constant — or omitted — collapses them, and an archived run then cannot be
    asked why the ledger holds two entries rather than one.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [
        _finding("correctness", line=11, title="off by one"),
        _finding("concurrency", line=13, title="a race on the same line")])]})
    _run(sourced_repo)
    findings = _artefact(capsys.readouterr().out)["findings"]

    assert len(findings) == 2, "signature dedup already collapsed them"
    assert len({(f["file"], f["line"] // 10, f["severity"]) for f in findings}) == 1, \
        "the fixture cannot express this axis: they differ by more than category"
    assert len({_signature(f) for f in findings}) == 2, \
        "the published record cannot tell two categories apart"


# --- 2. the verdict each persona actually cast -------------------------------

def test_each_persona_verdict_reaches_the_artefact_per_epoch(
        sourced_repo, monkeypatch, capsys):
    """You could not ask an archived run who voted YES."""
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding("correctness")])],
                        "Critic": [_contract("YES")],
                        "Sentinel": [_contract("YES")]})
    _run(sourced_repo)
    part = _artefact(capsys.readouterr().out)["participation"]

    assert all("verdict" in r for r in part), "a participation record casts no vote"
    assert {r["persona"]: r["verdict"] for r in part} == {
        "Analyst": "NO", "Critic": "YES", "Sentinel": "YES"}


def test_the_published_verdict_is_the_normalised_vote_not_the_prose(
        sourced_repo, monkeypatch, capsys):
    """A control value gets a two-word vocabulary; the prose stays where it is.

    ``normalise_verdict`` resolves exactly ``YES`` and ``NO`` and nothing else, so
    a hedge is ``null`` here and verbatim under ``unresolved_verdicts`` (#26).
    Publishing the raw string instead would put a second unbounded model-authored
    value in the record and let a reader match on prose.
    """
    _stub(monkeypatch, {"Analyst": [_contract("SOUND-WITH-CONCERNS")]},
          default=_contract("yes."))
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)
    by_persona = {r["persona"]: r["verdict"] for r in payload["participation"]}

    assert by_persona["Analyst"] is None, "a hedge is not a vote"
    assert by_persona["Critic"] == "YES", "'yes.' is a vote, normalised"
    assert [u["raw"] for u in payload["unresolved_verdicts"]] == \
        ["SOUND-WITH-CONCERNS"], "the prose left the record it belongs in"


def test_the_three_ways_a_verdict_is_absent_stay_distinguishable(
        sourced_repo, monkeypatch, capsys):
    """``null`` means three things, and this field does not collapse them.

    A persona whose call errored, one whose reply was unreadable, and one that
    reviewed and wrote a verdict that was not a vote all publish ``verdict:
    null``. Which it was is read from the siblings — ``status`` and
    ``unresolved_verdicts`` — never from this field, because a categorical value
    must not be recovered by matching prose (#24, #26, #30).
    """
    _stub(monkeypatch, {
        "Analyst": [CallResult("", 0, error="agent error: transport failed")],
        "Critic": [CallResult("nothing structured here", 0)],
        "Sentinel": [_contract("MAYBE", [_finding("ruin", severity="NOTE")])]},
        formatter=_NO_JSON)
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)
    rows = {r["persona"]: r for r in payload["participation"]}
    unread = {u["persona"] for u in payload["unresolved_verdicts"]}

    assert all(rows[n]["verdict"] is None for n in ("Analyst", "Critic", "Sentinel"))
    assert rows["Analyst"]["status"] == "agent_error"
    assert rows["Critic"]["status"] == "unreadable"
    assert rows["Sentinel"]["status"] in ("contributed", "found_nothing")
    assert unread == {"Sentinel"}, \
        "only the persona that WROTE an unreadable verdict is recorded as such"


def test_the_agreement_count_is_reconstructable_from_the_verdicts(
        sourced_repo, monkeypatch, capsys):
    """#82 published the count; this publishes the votes it was taken over.

    Exactly knowable and two-ended: ``yes_votes`` must equal the final-epoch rows
    whose verdict is a vote and whose status is not one the run knows produced no
    review, and no persona that did not review may carry a verdict at all.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding("correctness")])]})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)
    final = [r for r in payload["participation"] if r["epoch"] == payload["epochs"]]
    did_not_review = {"agent_error", "spawn_failed", "not_spawned", "unreadable"}

    counted = [r for r in final
               if r["verdict"] == "YES" and r["status"] not in did_not_review]
    assert payload["agreement"]["yes_votes"] == len(counted)
    assert all(r["verdict"] is None for r in final if r["status"] in did_not_review), \
        "a persona the run knows did not review is recorded as having voted"


# --- 3. the number the stall counter is taken on -----------------------------

def test_the_artefact_records_what_each_epoch_raised(
        sourced_repo, monkeypatch, capsys):
    """The three counts, and the threshold the last of them is read against.

    The fixture makes all three numbers DIFFERENT in one epoch, which is the only
    way a record carrying three counts can be shown to carry three: a BLOCKER and
    a NOTE at distinct signatures give two new findings, two new clusters, and one
    of them material. A fixture where they coincide cannot express the axis — a
    mutation reporting ``new_clusters`` as the material count survived it.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [
        _finding("correctness", line=10),
        _finding("style", line=50, severity="NOTE", title="a passing remark")])]})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)
    raised = payload["raised"]

    assert raised["stall_patience"] == 1, "the threshold the count is read against"
    assert len(raised["epochs"]) == payload["epochs"]
    (first,) = raised["epochs"]
    assert first == {"epoch": 1, "new_findings": 2, "new_clusters": 2,
                     "material_new_clusters": 1}, \
        "the three counts are not three separately-computed numbers"


def test_the_gate_prints_the_acted_on_count_beside_the_raw_one(
        sourced_repo, monkeypatch, capsys):
    """Both numbers, never one instead of the other (#73's precedent).

    The two differing is the fact an operator needs, so replacing one with the
    other would trade a contradiction for a different blind spot.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_finding("correctness")])]})
    _run(sourced_repo, "--max-epochs", "2", "--stall-patience", "2")
    lines = _synthesis_lines(capsys.readouterr().out)

    assert lines, "the gate printed no synthesis at all"
    assert lines[0].startswith("new findings: 1 | new material issues: 1 | "), \
        f"the gate line does not carry both numbers: {lines[0]!r}"


def test_a_divergent_epoch_prints_both_numbers(sourced_repo, monkeypatch, capsys):
    """The console half of the defect, on the only epoch that can show it.

    Under the default ``--stall-patience 1`` the divergent epoch IS the halting
    epoch, and ``human_gate`` runs only when the run does not halt — so no
    synthesis is printed for it at all. Patience 2 makes epoch 2 non-halting, and
    a MERGING reduce is required: under the identity default every new finding is
    a new cluster and the two numbers can never differ.
    """
    reworded = _contract("NO", [_finding("correctness", line=30,
                                         title="the same concept, re-worded")])
    _stub(monkeypatch,
          {"Analyst": [_contract("NO", [_finding("correctness")]), reworded]},
          cluster=CallResult(
              'Grouping.\n\n```json\n{"clusters": [[0, 1]]}\n```', 0))
    _run(sourced_repo, "--max-epochs", "3", "--stall-patience", "2",
         "--semantic-dedup")
    out = capsys.readouterr().out
    lines = _synthesis_lines(out)

    assert len(lines) >= 2, f"epoch 2 printed no synthesis: {lines!r}"
    assert lines[1].startswith("new findings: 1 | new material issues: 0 | "), \
        f"the epoch-2 gate line hides the number acted on: {lines[1]!r}"
    payload = _artefact(out)
    raised = payload["raised"]["epochs"][1]
    assert (raised["new_findings"], raised["material_new_clusters"]) == (1, 0), \
        "the artefact does not carry the divergence the console now shows"
    # The threshold this run was actually given, not the default. #82's precedent:
    # a run that publishes the value a gate reads publishes what it is read
    # against — and a patience reported as a constant survives every fixture that
    # only ever runs at the default one.
    assert payload["raised"]["stall_patience"] == 2, \
        "the artefact reports a patience the run was not given"
