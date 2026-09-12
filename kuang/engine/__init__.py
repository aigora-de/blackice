# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The deterministic engine: the bounded, human-convened adversarial review loop.

The public surface of the engine, re-exported so importers depend on
``kuang.engine`` rather than on which sibling module currently holds a name.

The engine knows nothing about any agent runtime: everything it needs arrives
through the ``Protocol`` seams in ``protocols``. That is asserted, not merely
asserted-in-prose — see ``tests/engine/test_backend_agnostic.py``.
"""

from .findings import (AFFIRMATIVE_VERDICT, DIAGNOSIS_BOUND, Cluster,
                       EpochResult, Finding, PersonaReport, PersonaStatus,
                       ReviewRun, Severity, Suppression, SurfaceFailure,
                       bounded_diagnosis)
from .halting import HaltingSet, HaltReason
from .loop import PanelConfig, ReviewSpec, run
from .protocols import (Adjudicate, GateDecision, GatherSurface, HumanGate,
                        Reduce, ReviewSurface, SpawnPersona)

__all__ = [
    "AFFIRMATIVE_VERDICT", "Adjudicate", "Cluster", "DIAGNOSIS_BOUND",
    "EpochResult", "Finding",
    "GateDecision", "GatherSurface", "HaltReason", "HaltingSet", "HumanGate",
    "PanelConfig", "PersonaReport", "PersonaStatus", "Reduce", "ReviewRun",
    "ReviewSpec",
    "ReviewSurface", "Severity", "SpawnPersona", "Suppression",
    "SurfaceFailure",
    "bounded_diagnosis", "run",
]
