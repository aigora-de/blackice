# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A finding the ledger drops to a collision must be reported (#119).

``Finding.key`` hashes four values joined with ``|``, two of them model-authored,
so two findings whose ``key_components`` DIFFER can hash alike — and ``loop.run``
keeps only the first. The second persona's claim then reached no ledger, no
artefact and no count, and **nothing anywhere said so**: a collision was
indistinguishable from the coarse dedup working exactly as designed.

The discriminator is exact and needs no encoding change: two findings that share a
key are the same finding **iff the tuple that was hashed is identical**. Compare
the components, never the digests — the digest is precisely what cannot tell the
two cases apart. That covers the class rather than one route: the unescaped
separator, the ``None``/``"None"`` route, and a truncated-digest accident all
report the same way.

These tests hold the **engine** half. The record is written where the ledger
decides (``loop.run``) rather than recomputed by a reporter, because the run is
handed to ``HumanGate`` mutably and a gate that removes a ledger key would make a
derived walk report a suppression that never happened — see
``test_a_gate_that_deletes_a_ledger_key_produces_no_false_suppression``, which is
that rejected option's guard.

Stdlib + ``FakeEnsemble`` only — no network, no ``claude`` subprocess.

**Scope, stated so the silence is not read as a claim.** Reporting is this
issue; the ambiguous encoding itself is **#125**, and two tests here exist to go
red there rather than now. Findings whose components are IDENTICAL are
re-sightings by the ledger's identity rule and are counted as such, including
every instrument diagnosis — that is **#126**, with its own live test below.
"""

from __future__ import annotations

from kuang.engine import (Finding, GateDecision, HaltingSet, PanelConfig,
                          PersonaReport, ReviewSpec, Severity, run)
from kuang.engine.fakes import FakeEnsemble

# Titles no other line, persona or path can carry, so "the suppressed one is
# named" cannot be satisfied by the finding that was KEPT, and an absence
# assertion cannot pass by accident (#87). The two findings in every collision
# fixture below are deliberately distinguishable by title alone.
_KEPT = "ZQX-holder-3180"
_LOST = "ZQX-suppressed-9042"


def _f(persona, title, *, claim_class="correctness", file="a.py", line=12,
       severity=Severity.BLOCKER, about_run=False, evidence=""):
    return Finding(persona=persona, title=title, severity=severity,
                   claim_class=claim_class, file=file, line=line,
                   evidence=evidence, about_run=about_run)


# The separator route, verified by execution before it was written down: the two
# differ in BOTH model-authored components and render to one hashed string.
def _holder() -> Finding:
    return _f("A", _KEPT, file="a.py|1", claim_class="correctness")


def _colliding() -> Finding:
    return _f("B", _LOST, file="a.py", claim_class="1|correctness")


def _report(persona, *findings, verdict="NO") -> PersonaReport:
    return PersonaReport(persona=persona, verdict=verdict, findings=list(findings))


def _panel(script, *, personas=("A", "B"), max_epochs=2, patience=1, **kwargs):
    return run(
        ReviewSpec(why="w", what="x"),
        HaltingSet(max_epochs=max_epochs, stall_patience=patience,
                   require_scope_complete=False),
        PanelConfig(personas=[(p, "m") for p in personas]),
        spawn=FakeEnsemble(script), gather=lambda e: "s", parallel=False,
        **kwargs)


def _titles(epoch) -> list[str]:
    return [s.finding.title for s in epoch.suppressed]


# --- 1. a collision is recorded, on every route ------------------------------

def test_a_collision_records_the_claim_the_ledger_refused():
    """The defect this issue is titled for, on the separator route.

    The collision is raised in epoch 1 of a two-epoch run whose final epoch is
    clean, deliberately: a fixture whose only collision lands in the FINAL epoch
    cannot tell a record of every epoch from one that reads the last, and a
    mutation doing the latter would survive it.
    """
    holder, lost = _holder(), _colliding()
    assert holder.key == lost.key, "the fixture must exercise one key"
    assert holder.key_components != lost.key_components, \
        "the fixture must be a collision, not a re-sighting"

    review = _panel({1: {"A": _report("A", holder), "B": _report("B", lost)}})

    assert _titles(review.epochs[0]) == [_LOST], \
        "the claim the ledger refused is not named"
    assert review.epochs[0].suppressed[0].holder.title == _KEPT, \
        "the record does not say which finding held the signature"
    assert review.epochs[1].suppressed == [], \
        "a clean epoch reports a suppression"


def test_the_none_route_collides_without_a_separator_and_is_recorded():
    """`f"{None}"` is `"None"`, and a persona may write that string.

    The second route, and the reason the comparison is over RAW components: a
    finding with no file and one whose file is the string ``None`` render to the
    same hashed string, so a diagnosis that compared the rendered values would
    miss this route entirely while passing every separator test above.
    """
    holder = _f("A", _KEPT, file=None, line=None)
    lost = _f("B", _LOST, file="None", line=None)
    assert holder.key == lost.key, "the fixture must exercise one key"
    assert (holder.key_components[0], lost.key_components[0]) == (None, "None"), \
        "the fixture cannot express the axis: the files must differ as None vs 'None'"

    review = _panel({1: {"A": _report("A", holder), "B": _report("B", lost)}})

    assert _titles(review.epochs[0]) == [_LOST]


def test_a_collision_in_a_later_epoch_is_recorded_against_that_epoch():
    """The ledger persists across epochs, so the drop can happen in any of them."""
    review = _panel({1: {"A": _report("A", _holder())},
                     2: {"B": _report("B", _colliding())}})

    assert review.epochs[0].suppressed == [], "epoch 1 dropped nothing"
    assert _titles(review.epochs[1]) == [_LOST]


# --- 2. the mirror image: what is NOT a collision -----------------------------

def test_a_legitimate_re_sighting_is_counted_and_never_reported_as_suppressed():
    """The coarse dedup working as designed must not read as a defect.

    A rule that fires on a healthy run is the mirror image of the defect it is
    meant to catch. Two personas raising the same concept at lines 12 and 15 —
    ONE bucket, identical components — are the same finding by the ledger's own
    identity rule, and the run says so as a count rather than a loss.

    The line moves within the bucket on purpose: a fixture whose re-sighting
    repeats the line exactly cannot see a mutation that compares the raw ``line``
    instead of the bucket.

    **The suppression half is GREEN on main + stubs and is a guard**, against the
    rejected rule that compares digests instead of components; the ``resighted``
    half is the regression test.
    """
    first = _f("A", _KEPT, line=12)
    again = _f("B", "the same concept, seen again", line=15)
    assert first.key == again.key, "the fixture must exercise one key"
    assert first.key_components == again.key_components, \
        "the fixture must be a re-sighting, not a collision"

    review = _panel({1: {"A": _report("A", first), "B": _report("B", again)}})

    assert review.epochs[0].suppressed == [], \
        "the coarse dedup working as designed was reported as a lost claim"
    assert review.epochs[0].resighted == 1


def test_two_instrument_diagnoses_are_a_re_sighting_and_not_a_collision():
    """LIVE SIBLING of #126, which this issue deliberately does not close.

    Every diagnosis site builds a finding with no file, no line, ``claim_class``
    ``"meta"`` at ``NOTE``, so all of them share one key and a run publishes only
    the first. Their components are IDENTICAL, so that loss is a re-sighting by
    the ledger's identity rule and this issue's report is correctly silent about
    it — it is a different defect with its own acceptance.

    This test is what stops #119's fix being read as covering it: it goes red at
    **#126**, by design, and until then it pins that the loss is counted rather
    than reported as a collision. The ``resighted`` half is the regression test;
    the suppression half is a guard.
    """
    a = _f("A", "agent error: transport failed", claim_class="meta",
           file=None, line=None, severity=Severity.NOTE, about_run=True)
    b = _f("B", "unparseable JSON findings: line 1", claim_class="meta",
           file=None, line=None, severity=Severity.NOTE, about_run=True)
    assert a.key_components == b.key_components, "the fixture must exercise one key"

    review = _panel({1: {"A": _report("A", a), "B": _report("B", b)}})

    assert review.epochs[0].suppressed == [], \
        "an identical-component merge was reported as a collision"
    assert review.epochs[0].resighted == 1
    assert len(review.ledger) == 1, "#126: the second diagnosis is still dropped"


def test_a_refuted_finding_is_neither_suppressed_nor_counted():
    """Adjudication runs BEFORE the insert, so a refuted claim never reaches it.

    ``loop.run`` filters on ``counts_open`` before the ledger, and the record
    must mirror that exactly or it reports a loss the run never suffered — a
    withdrawal on the evidence is not a dropped claim.

    **GREEN on main + stubs: a guard**, against a record written before the
    filter rather than after it.
    """
    review = _panel({1: {"A": _report("A", _holder()), "B": _report("B", _colliding())}},
                    adjudicate=lambda f, surface: f.title != _LOST)

    assert review.epochs[0].suppressed == [], "a refuted claim is not a lost one"
    assert review.epochs[0].resighted == 0
    assert len(review.ledger) == 1


def test_a_gate_that_deletes_a_ledger_key_produces_no_false_suppression():
    """The measured reason this is recorded at the insert, not derived after it.

    ``HumanGate`` receives the run mutably — the engine's own ``fakes._demo``
    gate rewrites ledger entries — so a reporter walking the epochs against the
    FINAL ledger can be wrong about the past: here epoch 1's finding **entered**
    the ledger, the gate removed it, and epoch 2's different finding took the
    same key. A derived walk reports epoch 1's claim as suppressed by a finding
    that did not exist when it was inserted.

    **GREEN on main + stubs, and it is the guard on the REJECTED option** (a
    ``ReviewRun`` property computing this after the fact), not a regression test.
    The mutation implementing that option turns it red.
    """
    holder, lost = _holder(), _colliding()

    def gate(result, review_run):
        review_run.ledger.pop(holder.key, None)
        return GateDecision(stop=False)

    review = _panel({1: {"A": _report("A", holder)},
                     2: {"B": _report("B", lost)}},
                    human_gate=gate, max_epochs=2)

    assert review.epochs[0].suppressed == [], "the gate deleted it; nothing dropped it"
    assert review.epochs[1].suppressed == [], \
        "epoch 2 inserted its finding into a ledger that no longer held the key"
    assert [f.title for f in review.epochs[1].new_findings] == [_LOST]


# --- 3. the three counts partition what the epoch emitted ---------------------

def test_every_emitted_finding_is_new_resighted_or_suppressed():
    """The partition, and the vacuity-forbidding partner of the tests above.

    Every open finding an epoch's reports carry is inserted, recognised as one
    the ledger already holds, or dropped by a collision. A run reporting only the
    first leaves the other two indistinguishable, which is this issue — and the
    equality is what lets ``check_invariants.py`` check the three counts against
    the per-persona totals rather than take them on trust (#61's rule, one layer
    down).

    The fixture makes all three non-zero at once, which is the only way a record
    carrying three numbers can be shown to carry three.
    """
    review = _panel({1: {"A": _report("A", _holder()),
                         "B": _report("B", _colliding()),
                         "C": _report("C", _f("C", "the same concept, seen again",
                                              file="a.py|1", line=15))}},
                    personas=("A", "B", "C"))
    epoch = review.epochs[0]
    emitted = sum(len(r.findings) for r in epoch.reports)

    assert (len(epoch.new_findings), epoch.resighted, len(epoch.suppressed)) == (1, 1, 1), \
        "the three counts are not three separately-computed numbers"
    assert emitted == (len(epoch.new_findings) + epoch.resighted
                       + len(epoch.suppressed)), \
        "a finding was emitted and is in none of the three"


# --- 4. reporting only: what this issue deliberately does not change ----------

def test_the_dedup_signature_is_unchanged_by_this_issue():
    """The digests, pinned as literals, because the fix must move none of them.

    ``key`` is now derived from ``key_components`` so the tuple a collision is
    diagnosed on and the tuple that was hashed cannot drift apart. That refactor
    is required to be invisible: an archived run's signatures still reconstruct
    with today's formula.

    **GREEN on main: a guard against re-encoding inside a reporting PR**, not a
    regression test. It goes red at **#125**, where the encoding changes on
    purpose, and it is to be updated there.
    """
    assert _holder().key == "0f1accdf46e2"
    assert _colliding().key == "0f1accdf46e2"
    assert _f("A", _KEPT, file=None, line=None).key == "c3f609b0d463"
    assert _f("B", _LOST, file="None", line=None).key == "c3f609b0d463"


def test_a_collision_still_drops_the_claim_from_the_ledger():
    """LIVE SIBLING of #125: this issue reports the loss, it does not stop it.

    The claim stays out of the ledger, so it counts toward no open-blocker total
    and does not reset the stall counter. Pinned so the two halves cannot be
    confused: a PR that quietly re-encoded the signature would turn this red.

    **GREEN on main: a guard**, and red at #125 by design.
    """
    review = _panel({1: {"A": _report("A", _holder()), "B": _report("B", _colliding())}})

    assert len(review.ledger) == 1, "the ledger kept both, so #125 landed here"
    assert [f.title for f in review.epochs[0].new_findings] == [_KEPT]
    assert len(review.open_blockers) == 1, \
        "a suppressed claim must not be counted as though it were kept"
