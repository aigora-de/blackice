# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A run must say which claim its ledger dropped to a collision (#119).

Two findings whose hashed components differ can share a ``Finding.key``, and the
ledger keeps the first. Until this issue the second persona's claim vanished
without a word anywhere: not in the artefact, not on the console, not in any
count — and a collision read exactly like the coarse dedup working as intended.

The channel decision, recorded here because it is the substance:

* the **artefact** carries a top-level ``suppressed`` array, unconditional, each
  entry naming the dropped claim with the components that were hashed. Naming
  rather than counting, because the point of a crafted collision is to delete a
  claim — a count leaves the deletion successful on the record — and because a
  human adjudicates from this artefact and cannot adjudicate what they cannot
  read;
* the **console** prints a section only when something was dropped, following
  #24's and #26's exception sections: on a healthy run its absence is itself a
  complete claim;
* **not** ``report.ledger_line`` — cross-epoch memory and the ``--prior-findings``
  seed share that renderer, so a notice there enters the next epoch's prompt
  (#71, #63) — **not** a ``meta`` finding, which would enter the ledger and add to
  ``issues_about_run`` (#73's defect), and **not** the gate synthesis, which is
  #117's channel.

These drive ``kuang.cli.main`` end to end over a throwaway repo with the
subprocess boundary stubbed — nothing spawned, no network.

Console assertions are anchored to the line they are about, never searched across
the whole output (#87, #103).
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code import session as session_module
from kuang.backends.claude_code.spawn import CallResult
from kuang.cli import EVIDENCE_BOUND, main

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

# Values no title, persona, path or category can collide with, so "the suppressed
# claim is named" cannot be satisfied by the one that was KEPT, and "this never
# reached that channel" cannot pass by accident (#87).
_KEPT = "ZQX-holder-3180"
_LOST = "ZQX-suppressed-9042"
_EVIDENCE = "ZQX-working-5561"


def _contract(verdict="YES", findings=()) -> CallResult:
    body = json.dumps({"verdict": verdict, "findings": list(findings)})
    return CallResult(f"I reviewed it.\n\n```json\n{body}\n```\n", 0)


def _finding(*, title, claim_class="correctness", file="a.py", line=12,
             severity="BLOCKER", evidence="read it"):
    """One finding as a persona writes it. ``file=None`` omits the key."""
    f = {"title": title, "severity": severity, "claim_class": claim_class,
         "evidence": evidence}
    if file is not None:
        f["file"] = file
    if line is not None:
        f["line"] = line
    return f


# The separator route: the two differ in BOTH model-authored components and
# render to one hashed string, so the ledger keeps only the first.
def _holder(**kw):
    return _finding(title=_KEPT, file="a.py|1", claim_class="correctness", **kw)


def _colliding(**kw):
    return _finding(title=_LOST, file="a.py", claim_class="1|correctness", **kw)


_CLEAN = _contract()
_NO_JSON = CallResult("I read the diff and have nothing structured to say.", 0)


@pytest.fixture
def sourced_repo(changed_repo):
    (changed_repo / "CLAUDE.md").write_text(_CLAUDE_MD)
    return changed_repo


def _stub(monkeypatch, replies: dict, *, default=_CLEAN, formatter=_NO_JSON) -> list[str]:
    """Reply per persona per epoch; returns the list of prompts as sent.

    The prompts are captured because one of this issue's refusals is about them:
    a suppressed claim must not reach the next epoch's panel through cross-epoch
    memory, and the only honest way to assert that is to read what was sent.
    """
    seen: dict[str, int] = {}
    prompts: list[str] = []

    def _fake(self, prompt, mandate, tools, model):  # noqa: ANN001, ARG001
        prompts.append(prompt)
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
    return prompts


def _run(repo, *extra) -> int:
    return main(["--repo", str(repo), "--base", "HEAD~1", "--max-epochs", "1",
                 "--no-parallel", *extra])


def _artefact(out: str) -> dict:
    return json.loads(out.split("--- JSON ---")[-1])


def _console(out: str) -> str:
    """The human-readable half only: everything before the artefact block."""
    return out.split("--- JSON ---")[0]


_HEADER = "suppressed findings:"


def _suppression_block(out: str) -> list[str]:
    """The console section's lines: its header and everything indented under it.

    Anchored to the header rather than searched for across the console, because
    a title, a path or a category can carry any word this section uses (#87).
    """
    lines = _console(out).splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(_HEADER)]
    if not starts:
        return []
    block = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        if not line.startswith("  "):
            break
        block.append(line)
    return block


# --- 1. the artefact names the claim that was dropped -------------------------

def test_the_artefact_names_the_claim_the_ledger_dropped(
        sourced_repo, monkeypatch, capsys):
    """The defect this issue is titled for, end to end through the CLI.

    A two-epoch run whose collision is in epoch 1 and whose second epoch is
    clean: a fixture colliding only in the FINAL epoch cannot tell a record of
    every epoch from one that reads the last.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()]), _CLEAN],
                        "Critic": [_contract("NO", [_colliding()]), _CLEAN]})
    _run(sourced_repo, "--max-epochs", "2")
    payload = _artefact(capsys.readouterr().out)

    assert [s["title"] for s in payload["suppressed"]] == [_LOST], \
        "the run does not say which claim its ledger refused"
    (lost,) = payload["suppressed"]
    assert (lost["persona"], lost["epoch"], lost["severity"]) == ("Critic", 1, "BLOCKER")
    # The components AS HASHED, so a reader can recompute the signature and see
    # that this is a collision rather than a re-sighting.
    assert (lost["file"], lost["line"], lost["claim_class"]) == \
        ("a.py", 12, "1|correctness")
    assert [f["title"] for f in payload["findings"]] == [_KEPT], \
        "the ledger kept both, so this is no longer a collision"


def test_a_healthy_run_publishes_an_empty_suppression_record(
        sourced_repo, monkeypatch, capsys):
    """Unconditional, because an absent key makes no claim (#103's precedent).

    The mirror image as well: two personas raising ONE concept is the coarse
    dedup working as designed, and a run that reported it as a lost claim would
    have this issue's defect inverted. The re-sighting is counted instead.
    """
    again = _finding(title="the same concept, seen again", file="a.py|1", line=15)
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()])],
                        "Critic": [_contract("NO", [again])]})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert payload["suppressed"] == [], \
        "the coarse dedup working as designed was published as a lost claim"
    assert payload["raised"]["epochs"][0]["resighted"] == 1, \
        "a run that merged two findings does not say it merged them"


def test_a_suppressed_claims_evidence_is_bounded_like_a_published_finding(
        sourced_repo, monkeypatch, capsys):
    """One persona reply may not put an unbounded string into a shared artefact.

    The tail is a sentinel rather than more filler: a homogeneous string's tail
    is a substring of its own head, so "the tail was cut" would pass whatever the
    code did.
    """
    raw = "A" * EVIDENCE_BOUND + "-tail-" + _EVIDENCE
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()])],
                        "Critic": [_contract("NO", [_colliding(evidence=raw)])]})
    _run(sourced_repo)
    (lost,) = _artefact(capsys.readouterr().out)["suppressed"]

    assert lost["evidence_chars"] == len(raw), \
        "the record must say how long the evidence was BEFORE the bound"
    assert lost["evidence"].startswith(raw[:EVIDENCE_BOUND])
    assert raw[EVIDENCE_BOUND:] not in lost["evidence"], "the tail was not cut"
    assert len(lost["evidence"]) > EVIDENCE_BOUND, \
        "the marker rides on TOP of the bound, never out of it"


# --- 2. the console says so, and only when there is something to say ----------

def test_the_console_reports_the_dropped_claim_and_what_held_its_signature(
        sourced_repo, monkeypatch, capsys):
    """An operator reading only the human-readable half must see the loss.

    Both sides are printed with their own components, because that is what makes
    a collision legible AS a collision — the two findings agree on severity and
    line bucket, and differ exactly where the encoding let them merge.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()])],
                        "Critic": [_contract("NO", [_colliding()])]})
    _run(sourced_repo)
    block = _suppression_block(capsys.readouterr().out)

    assert block, "the console says nothing about a claim the run dropped"
    assert block[0].startswith(f"{_HEADER} 1"), \
        f"the header does not count the dropped claims: {block[0]!r}"
    entry, held = block[1], block[2]
    assert "(epoch 1)" in entry and "[BLOCKER]" in entry
    assert "(Critic)" in entry and _LOST in entry, \
        f"the dropped claim is not named on its own line: {entry!r}"
    assert "'a.py':12" in entry and "1|correctness" in entry, \
        f"the line omits the components that were hashed: {entry!r}"
    assert "(Analyst)" in held and _KEPT in held, \
        f"the line does not say what held the signature: {held!r}"
    assert "'a.py|1':12" in held and "correctness" in held


def test_the_console_can_tell_the_none_route_apart_from_a_re_sighting(
        sourced_repo, monkeypatch, capsys):
    """The route with no separator, and the reason locations print with ``repr``.

    A finding with no file and one whose file is the string ``None`` hash alike.
    Rendered as ``str`` both print ``None:None``, so the section would show two
    identical locations and read exactly like the re-sighting it is not — the
    console failing at the one job this section has.
    """
    holder = _finding(title=_KEPT, file=None, line=None)
    lost = _finding(title=_LOST, file="None", line=None)
    _stub(monkeypatch, {"Analyst": [_contract("NO", [holder])],
                        "Critic": [_contract("NO", [lost])]})
    _run(sourced_repo)
    block = _suppression_block(capsys.readouterr().out)

    assert block, "the console says nothing about a claim the run dropped"
    entry, held = block[1], block[2]
    assert "'None':None" in entry, \
        f"the dropped claim's file is not distinguishable from an absent one: {entry!r}"
    assert "None:None" in held and "'None'" not in held, \
        f"the holder's absent file is rendered as though it were the string: {held!r}"


def test_a_suppressed_ruin_class_claim_is_marked_on_the_console(
        sourced_repo, monkeypatch, capsys):
    """The breaker never saw it, and the operator is told so in those terms.

    On this route the holder is an UGLY too, so the breaker latches anyway. The
    mark exists for the route where it does not: a truncated-digest collision
    puts no constraint on severity, so an UGLY can be dropped by a lower-severity
    holder and the run halt without escalating. That is #125's to close; this
    says it out loud wherever it happens.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder(severity="UGLY")])],
                        "Critic": [_contract("NO", [_colliding(severity="UGLY")])]})
    _run(sourced_repo)
    block = _suppression_block(capsys.readouterr().out)

    assert block, "the console says nothing about a claim the run dropped"
    assert "UGLY" in block[1] and "breaker" in block[1], \
        f"a ruin-class claim was dropped without a word about the gate: {block[1]!r}"


def test_a_healthy_run_prints_no_suppression_section(
        sourced_repo, monkeypatch, capsys):
    """The section's absence is itself a complete claim (#24, #26).

    **GREEN on main: a guard** against an always-on section, which would print a
    heading about lost claims on every clean run — this defect inverted, and the
    shape #103's close-out warned about.
    """
    again = _finding(title="the same concept, seen again", file="a.py|1", line=15)
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()])],
                        "Critic": [_contract("NO", [again])]})
    _run(sourced_repo)

    assert _suppression_block(capsys.readouterr().out) == [], \
        "a run that dropped nothing printed a section about dropped claims"


# --- 3. the channels this deliberately does not reach -------------------------

def test_the_dropped_claim_reaches_the_artefact_and_no_other_channel(
        sourced_repo, monkeypatch, capsys):
    """The channel decision, asserted rather than described.

    Presence first, so this cannot pass vacuously: on ``main`` the claim is
    dropped entirely, and every absence below would hold for the wrong reason.
    Then the two refusals — the persona's own working stays out of the console,
    and the dropped claim stays out of the next epoch's prompt, which
    ``epoch_summary`` and ``load_prior_findings`` share (#112's third channel).
    """
    prompts = _stub(monkeypatch,
                    {"Analyst": [_contract("NO", [_holder()]), _CLEAN],
                     "Critic": [_contract("NO", [_colliding(evidence=_EVIDENCE)]),
                                _CLEAN]})
    _run(sourced_repo, "--max-epochs", "2")
    out = capsys.readouterr().out

    (lost,) = _artefact(out)["suppressed"]
    assert lost["evidence"] == _EVIDENCE, "the record keeps the claim but not the check"
    assert _EVIDENCE not in _console(out), \
        "model prose about the surface reached the console"
    assert not any(_LOST in p for p in prompts), \
        "the dropped claim reached a later panel's prompt"
    assert len(prompts) > 3, "the fixture never ran a second epoch to carry memory"


def test_a_dropped_claim_is_not_a_finding_about_the_run(
        sourced_repo, monkeypatch, capsys):
    """Not a ``meta`` finding: that would enter the ledger it was dropped from.

    #73's defect exactly — a failure of the instrument counted among the defects
    in the change. It would also add to ``issues_about_run`` and, being itself a
    locationless ``meta`` NOTE, could collide with the next one.

    **GREEN on main: a guard on the rejected option**, not a regression test.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()])],
                        "Critic": [_contract("NO", [_colliding()])]})
    _run(sourced_repo)
    payload = _artefact(capsys.readouterr().out)

    assert len(payload["findings"]) == 1, "the record was added to the ledger"
    assert payload["issues_about_run"] == 0, \
        "a dropped claim was counted as a failure of the instrument"
    assert payload["issues_in_change"] == 1


def test_the_gate_synthesis_says_nothing_about_suppression(
        sourced_repo, monkeypatch, capsys):
    """The gate is #117's channel, and the halting epoch never reaches it.

    A suppression notice there would be reported for some epochs and not others,
    with no way for a reader to tell which — so the console section at the end of
    the run carries every epoch, and the gate carries none.

    **GREEN on main: a guard on the rejected channel**, not a regression test.
    """
    _stub(monkeypatch, {"Analyst": [_contract("NO", [_holder()]), _CLEAN],
                        "Critic": [_contract("NO", [_colliding()]), _CLEAN]})
    _run(sourced_repo, "--max-epochs", "2")
    console = _console(capsys.readouterr().out)
    gate = console.split("=== epoch 1 synthesis ===")[-1].split("=== HALT")[0]

    assert _LOST not in gate, "the gate synthesis carried the dropped claim"
    assert _HEADER not in gate, "the suppression section printed at the gate"
