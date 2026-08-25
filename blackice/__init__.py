# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""blackice — a human-in-the-loop adversarial code-review panel.

A bounded, human-convened loop of reviewer personas — each tasked to *break* a
change, not approve it — with a ruin-class circuit-breaker. The panel informs; a
human adjudicates and decides.

Three concerns, three subpackages:

* ``blackice.engine``   — the deterministic loop: halting, dedup, budget, and the
                          UGLY circuit-breaker. Knows nothing about any runtime.
* ``blackice.backends`` — bindings to a particular agent runtime.
* ``blackice.cli``      — the entry point, the only place that knows both exist.

``python -m blackice`` runs it.
"""

__version__ = "0.1.0"
