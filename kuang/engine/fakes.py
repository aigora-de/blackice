# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Offline test doubles: a scripted ensemble and a reference end-to-end demo.

No network, no subprocess, no wall-clock — the loop can be exercised whole from
a dict. ``python -m kuang.engine.fakes`` runs the demo.
"""

from __future__ import annotations

from .findings import EpochResult, Finding, PersonaReport, ReviewRun, Severity
from .halting import HaltingSet, HaltReason
from .loop import PanelConfig, ReviewSpec, run
from .protocols import GateDecision, ReviewSurface


class FakeEnsemble:
    """A scripted ``SpawnPersona`` that replays canned per-epoch reports.

    ``script[epoch][persona]`` -> ``PersonaReport``. Missing entries yield an
    empty, YES-verdict report (a persona with nothing left to say).
    """

    def __init__(self, script: dict[int, dict[str, PersonaReport]]) -> None:
        self._script = script

    def __call__(
        self, persona: str, mandate: str, surface: ReviewSurface, epoch: int  # noqa: ARG002
    ) -> PersonaReport:
        return self._script.get(epoch, {}).get(
            persona, PersonaReport(persona=persona, verdict="YES")
        )


def _demo() -> ReviewRun:
    """Tiny end-to-end demo: epoch 1 finds a BLOCKER, it is fixed, epoch 2 converges."""
    panel = PanelConfig(personas=[("quant", "find wrong maths"), ("engineer", "find bugs")])
    spec = ReviewSpec(why="records-critical path", what="the guard")
    halting = HaltingSet(max_epochs=5, stall_patience=1, require_scope_complete=False)

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
        # epoch 2: the blocker was fixed between epochs, both personas clear.
        2: {
            "quant": PersonaReport(persona="quant", verdict="YES"),
            "engineer": PersonaReport(persona="engineer", verdict="YES"),
        },
    }

    # Human gate marks the blocker resolved before epoch 2 by refuting it via
    # adjudication on re-look (here simulated: after epoch 1 we "fix" it, so on
    # epoch 2 nobody re-raises it and the ledger blocker is retired by the gate).
    fixed: dict[str, bool] = {}

    def gate(result: EpochResult, run: ReviewRun) -> GateDecision:
        for f in run.open_blockers:
            fixed[f.key] = True
            run.ledger[f.key] = Finding(**{**f.__dict__, "verified": False})  # resolved
        return GateDecision(stop=False, note="applied fix")

    review_run = run(
        spec, halting, panel,
        spawn=FakeEnsemble(script),
        gather=lambda epoch: f"surface@{epoch}",
        human_gate=gate,
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
