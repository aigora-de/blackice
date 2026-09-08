# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A surface that cannot be re-gathered must not discard the epochs that ran (#85).

``gather`` is called once per epoch, at the top of the loop, and nothing caught it.
So a surface that could not be reassembled on epoch 2 or later propagated out of
``run`` and took the ``ReviewRun`` with it — every finding, participation record,
surface record and token count from the epochs that **did** complete, for a panel
that had already been spawned and paid for.

The fix is the treatment ``spawn`` has had since #25, five lines below: the seam is
a fallible black box, the failure is caught as ``Exception`` (never a named backend
class, which the engine may not import, and never ``BaseException``, so a human's
Ctrl-C still stops the loop), and it is **recorded** rather than swallowed.

The asymmetry is the whole design. A failure with **no completed epoch** re-raises,
because there is nothing to report and #18's doctrine — a surface that cannot be
built is an operator error, never a review — is unchanged. The predicate is "no
epoch has completed", not "the counter is above one": the two coincide, but only
the first states the rule the reporter depends on.

**Which of these are regressions, said plainly, and measured rather than assumed.**
Of #85's nine tests — the two #111 added are labelled on themselves — seven go red
run against the ``main`` that preceded them, with this file in place. Six are the
defect: everything under "the defect", plus ``test_an_epoch_that_never_began_leaves_
no_record`` and ``test_a_lost_surface_pre_empts_the_epoch_ceiling``, both of which
raise today rather than halting. The seventh,
``test_a_healthy_run_never_reports_a_lost_surface``, goes red only for the missing
attribute: what it pins does not move, and it is the mirror-image guard rather than a
regression. The two that pass before and after are #18's path and Ctrl-C.

**One test that is deliberately absent.** "The breaker still outranks
``SURFACE_LOST``" names a state the code refuses to enter: an open UGLY escalates in
the epoch it appears, so the loop halts before a second epoch exists to lose a
surface in. ``test_unreviewed_panel.py`` records the identical trap for ``NO_REVIEW``
— asserting it would be a test killed by nothing.
"""

from __future__ import annotations

import pytest

from kuang.engine import (Finding, HaltingSet, HaltReason, PanelConfig,
                          PersonaReport, PersonaStatus, ReviewSpec, Severity, run)

PANEL = PanelConfig(personas=[(n, "mandate") for n in ("correctness", "adversary")])


class _BackendSurfaceError(RuntimeError):
    """Stands in for a backend's own exception class, which the engine may not name.

    Declared here rather than imported from ``kuang.backends`` on purpose: the
    engine's guard must be stated over ``Exception``, so a class it has never heard
    of has to work. ``tests/engine/test_backend_agnostic.py`` enforces the import
    side of the same rule.
    """


def _spawn(persona, mandate, surface, epoch):  # noqa: ANN001, ARG001
    """One persona, one open BLOCKER per epoch: material work worth not discarding.

    The finding is distinct per epoch — a different ``claim_class``, so a different
    ``Finding.key`` — because an identical one deduplicates into the ledger, resets
    nothing, and STALLs the run on epoch 2. That would stop the loop before any of
    these tests reached the state they name.
    """
    return PersonaReport(
        persona=persona, verdict="NO", tokens=9,
        status=PersonaStatus.CONTRIBUTED,
        findings=[Finding(persona, f"{persona}: off-by-one at epoch {epoch}",
                          Severity.BLOCKER, f"{persona}-lens-{epoch}", "a.py", 1)])


def _fails_on(epoch_to_fail: int, exc: BaseException | None = None):
    """A ``GatherSurface`` that assembles a surface until the named epoch, then cannot."""
    def gather(epoch: int) -> str:
        if epoch >= epoch_to_fail:
            raise exc or _BackendSurfaceError(
                "no reviewable files in the requested paths: a.py")
        return "def f():\n    return 1\n"
    return gather


def _run(gather, *, max_epochs: int = 3, checkpoint=None):
    return run(ReviewSpec(why="mission-critical", what="the diff"),
               HaltingSet(max_epochs=max_epochs), PANEL,
               spawn=_spawn, gather=gather, parallel=False, checkpoint=checkpoint)


# --- the defect ---------------------------------------------------------------

def test_a_failed_re_gather_returns_the_epochs_that_completed():
    """REGRESSION for #85: this raised out of ``run`` and the run was lost."""
    review_run = _run(_fails_on(2))

    assert review_run.halt_reason is HaltReason.SURFACE_LOST
    assert len(review_run.epochs) == 1, "the epoch that completed, and no phantom one"
    assert [f.title for f in review_run.open_blockers] == [
        "correctness: off-by-one at epoch 1",
        "adversary: off-by-one at epoch 1"], "the paid-for work survives"


def test_the_failure_names_the_epoch_it_failed_on():
    """The halt line cannot say it: it counts epochs that COMPLETED, so it says 1.

    An operator reading "surface_lost after 1 epoch(s)" has to be told that epoch 2
    is the one that could not be assembled, and why.
    """
    review_run = _run(_fails_on(2))

    assert review_run.surface_failure is not None
    assert review_run.surface_failure.epoch == 2
    assert review_run.surface_failure.detail == (
        "_BackendSurfaceError: no reviewable files in the requested paths: a.py")


def test_a_multi_line_message_is_recorded_as_one_line():
    """Measured, not assumed: a real diff-mode failure carries git's stderr.

    ``git diff no-such-ref...HEAD`` fails with a three-line diagnosis, and the
    backend's ``SurfaceError`` carries it whole. Printed as-is inside the report
    block, the continuation lines arrive unprefixed between the halt line and the
    findings and read as sections of the report. Collapsed here rather than at the
    console so that the artefact and the console carry the same string and cannot
    drift, which is why the bound lives here too.
    """
    review_run = _run(_fails_on(2, _BackendSurfaceError(
        "failed (128): fatal: ambiguous argument\nUse '--' to separate paths\n"
        "'git <command> [<revision>...]'")))

    assert "\n" not in review_run.surface_failure.detail
    assert review_run.surface_failure.detail == (
        "_BackendSurfaceError: failed (128): fatal: ambiguous argument Use '--' "
        "to separate paths 'git <command> [<revision>...]'")


def test_an_unbounded_message_is_bounded_after_it_is_collapsed():
    """A backend's exception text is model- or git-fed and has no length contract.

    The same 400-character bound ``spawn``'s guard applies to the exception it
    records (#25), for the same reason: this string is printed and written into an
    artefact meant to be shared.

    The ORDER is the second claim, and the message carries a long whitespace run so
    that it can be seen at all: a bound applied before the collapse spends itself on
    whitespace nobody will ever be shown, and 400 characters of diagnosis becomes
    23. A mutation swapping the two survived a version of this test written with a
    whitespace-free message — the matrix only tests the axis the input can express.
    """
    review_run = _run(_fails_on(2, _BackendSurfaceError("a" + " " * 500 + "b" * 500)))

    detail = review_run.surface_failure.detail
    kept, marker = detail.split("…", 1)
    assert len(kept) == 400
    assert marker.startswith(" [truncated:"), "the cut says so (#111)"
    assert kept.startswith("_BackendSurfaceError: a b"), "one space, not five hundred"
    assert kept.endswith("b"), "the bound spent on diagnosis, not on whitespace"


def test_a_cut_diagnosis_says_it_was_cut_and_records_what_it_was():
    """REGRESSION for #111, and the half a reader can see.

    ``surface_lost.detail`` is on the console and in the run artefact, and before
    this it was cut at 400 with nothing to say so. Measured, an ordinary path-mode
    refusal is cut at around ten unresolved paths and the cut lands **mid-path**,
    so the record ended on something that reads exactly like a path the tool failed
    to resolve.

    Two halves, deliberately. The marker is for the person reading it; nothing may
    read it back. ``detail_chars`` is the structural fact — the length of the
    collapsed diagnosis **before** bounding — so "was it cut" is ``detail_chars >
    400``, exactly knowable and immune to any rewording of the marker. That is the
    shape ``build_path_surface`` already uses one module over, where
    ``SurfaceRecord.truncated_file`` is returned rather than matched out of
    ``--- OMITTED ---``.
    """
    paths = ", ".join(f"src/module_{i}/component_handler_{i}.py" for i in range(1, 21))
    message = (f"no reviewable files in the requested paths: {paths}. Check they "
               "exist, are tracked by git, and are not gitignored.")
    review_run = _run(_fails_on(2, _BackendSurfaceError(message)))

    failure = review_run.surface_failure
    assert failure.detail_chars == len(f"_BackendSurfaceError: {message}"), (
        "the length of what there was to say, not of what was kept")
    assert failure.detail_chars > 400
    assert "truncated" in failure.detail
    assert not failure.detail.endswith(".py"), (
        "the record must not end on a path-shaped fragment an operator would read "
        "as a path that failed to resolve")


def test_a_short_diagnosis_carries_no_marker_and_says_its_own_length():
    """The MIRROR IMAGE, and the half that stops the guard passing vacuously.

    "A cut diagnosis says it was cut" is satisfied by marking every diagnosis,
    which would be this defect inverted — the record claiming a loss that never
    happened. So the pair: an uncut ``detail`` is byte-identical to the message,
    and ``detail_chars`` agrees with it rather than being a constant.

    ``detail_chars`` is present either way. An absent key makes no claim, and a
    reader coming to an artefact cold could not otherwise tell a diagnosis that
    fitted from one written before the field existed — the rule ``surface_lost``
    itself follows.

    **Red on main, but not a regression, and the distinction is the honest label.**
    It fails there only on the missing attribute: the first assertion — that a short
    diagnosis is byte-identical to the message — passes before and after, because
    nothing ever truncated it. What it PINS does not move. It is here as the mirror
    image of the test above, and it is killed by the vacuity mutation.
    """
    review_run = _run(_fails_on(2))

    failure = review_run.surface_failure
    expected = ("_BackendSurfaceError: no reviewable files in the requested "
                "paths: a.py")
    assert failure.detail == expected, "unchanged, not merely unmarked"
    assert failure.detail_chars == len(expected) <= 400


# --- the boundary: what this change deliberately does NOT touch ----------------

def test_a_first_epoch_with_no_surface_still_raises():
    """GUARD for #18, and the half of this fix that is an omission.

    No epoch completed, so there is nothing to report and no review happened. The
    operator error propagates exactly as it did before, the CLI reports it on
    stderr and prints no halt line, and the doctrine is unchanged. Were this to
    halt instead, the run would report an empty review as an outcome — #18
    inverted — and the reporter would ``IndexError`` on ``epochs[-1]`` reaching
    for a final epoch that does not exist.
    """
    with pytest.raises(_BackendSurfaceError):
        _run(_fails_on(1))


def test_ctrl_c_is_not_a_lost_surface():
    """GUARD: ``Exception``, never ``BaseException`` — #25's exclusion, restated.

    A human interrupting a run has stopped it; swallowing that into a halt reason
    would make the tool decide, and would make Ctrl-C unreliable at the one moment
    an operator most needs it.
    """
    with pytest.raises(KeyboardInterrupt):
        _run(_fails_on(2, KeyboardInterrupt()))


def test_a_healthy_run_never_reports_a_lost_surface():
    """MIRROR IMAGE: a rule that fires on a healthy run is the defect reversed.

    Three epochs, a surface every time. The halt reason is the ceiling it always
    was, the epoch count is unchanged, and the run makes no claim about a surface
    it did not lose.
    """
    review_run = _run(lambda epoch: "def f():\n    return 1\n")

    assert review_run.halt_reason is HaltReason.EPOCH
    assert len(review_run.epochs) == 3
    assert review_run.surface_failure is None


def test_an_epoch_that_never_began_leaves_no_record():
    """The question none of the other tests here asks: what the lost epoch leaves.

    Nothing. No ``EpochResult`` is appended for an epoch that never ran, so
    ``epochs`` stays the count of epochs that COMPLETED — which is what the halt
    line reports and what ``surface_failure.epoch`` exists to complement. The
    persistence hook is not fired for it either: ``checkpoint`` is documented as
    called each epoch for resumability, and an epoch with no surface, no reports and
    no findings has nothing to resume from. It would also rewrite the previous
    epoch's record with itself, which is a fact about the loop's control flow
    presenting as a fact about the review (#31 owns what that record should BE).
    """
    seen: list[int] = []
    review_run = _run(_fails_on(2), checkpoint=lambda r: seen.append(len(r.epochs)))

    assert seen == [1], "one checkpoint, for the one epoch that happened"
    assert len(review_run.epochs) == 1
    assert [e.index for e in review_run.epochs] == [1]


def test_a_lost_surface_pre_empts_the_epoch_ceiling():
    """Which reason the run stops for, when both are true on the same epoch.

    The surface is gone on epoch 3, which is also the ceiling. The run must say the
    reason it actually stopped rather than the one it would have reached anyway —
    the same ordering argument #72 made for ``NO_REVIEW``.
    """
    review_run = _run(_fails_on(3), max_epochs=3)

    assert review_run.halt_reason is HaltReason.SURFACE_LOST
    assert len(review_run.epochs) == 2
