# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Offline test doubles: a scripted ensemble and a reference end-to-end demo.

No network, no subprocess, no wall-clock — the loop can be exercised whole from
a dict. ``python -m kuang.engine.fakes`` runs the demo.
"""

from __future__ import annotations

from .findings import (Finding, PersonaReport, PersonaStatus,
                       ReviewRun, Severity)
from .halting import HaltingSet, HaltReason
from .loop import PanelConfig, ReviewSpec, run
from .protocols import GateDecision, ReviewSurface


class FakeEnsemble:
    """A scripted ``SpawnPersona`` that replays canned per-epoch reports.

    ``script[epoch][persona]`` -> ``PersonaReport``. Missing entries yield an
    empty, YES-verdict report (a persona with nothing left to say) — which is
    ``FOUND_NOTHING``, said explicitly, because "set the status at the source"
    (#30) is a rule about every ``SpawnPersona`` and a test double is one.
    """

    def __init__(self, script: dict[int, dict[str, PersonaReport]]) -> None:
        self._script = script

    def __call__(
        self, persona: str, mandate: str, surface: ReviewSurface, epoch: int  # noqa: ARG002
    ) -> PersonaReport:
        return self._script.get(epoch, {}).get(
            persona, PersonaReport(persona=persona, verdict="YES",
                                   status=PersonaStatus.FOUND_NOTHING)
        )


def _demo() -> ReviewRun:
    """Tiny end-to-end demo: epoch 1 raises a BLOCKER, adjudication refutes it, epoch 2
    converges.

    The demo used to "fix" the BLOCKER at the gate by rewriting ``run.ledger`` in place,
    which is how it converged. Since #155 no seam can change the run through an object it
    is handed, the gate included (the maintainer's ruling of 2026-10-14), so that write no
    longer reaches anything. A finding is now closed through a channel the engine owns:
    the adjudicator refutes it against source, "the author withdraws on the evidence".
    Retiring a finding a human FIXED is #157's, on the gate's return value.
    """
    panel = PanelConfig(personas=[("quant", "find wrong maths"), ("engineer", "find bugs")])
    spec = ReviewSpec(why="records-critical path", what="the guard")
    # Patience 2, not 1: a refuted finding never reaches the ledger, so epoch 1 adds no
    # material cluster, and patience 1 would halt it ``stall`` before epoch 2 converges.
    halting = HaltingSet(max_epochs=5, stall_patience=2, require_scope_complete=False)

    script = {
        1: {
            "quant": PersonaReport(
                persona="quant",
                verdict="NO",
                findings=[Finding("quant", "off-by-one in allocation",
                                  Severity.BLOCKER, "alloc", "x.py", 42)],
            ),
            "engineer": PersonaReport(persona="engineer", verdict="YES"),
        },
        # epoch 2: the blocker was refuted at epoch 1, both personas clear.
        2: {
            "quant": PersonaReport(persona="quant", verdict="YES"),
            "engineer": PersonaReport(persona="engineer", verdict="YES"),
        },
    }

    def adjudicate(finding: Finding, surface: ReviewSurface) -> bool:  # noqa: ARG001
        # The off-by-one does not reproduce against source: refuted, so dropped.
        return finding.claim_class != "alloc"

    review_run = run(
        spec, halting, panel,
        spawn=FakeEnsemble(script),
        gather=lambda epoch: f"surface@{epoch}",
        adjudicate=adjudicate,
        human_gate=lambda result, run_: GateDecision(stop=False, asked=True),
    )
    return review_run


if __name__ == "__main__":
    result = _demo()
    print(f"halt_reason = {result.halt_reason.value}")
    print(f"epochs      = {len(result.epochs)}")
    print(f"converged   = {result.converged}")
    assert result.halt_reason is HaltReason.CONVERGED, result.halt_reason
    assert len(result.epochs) == 2, len(result.epochs)
    print("demo OK")
