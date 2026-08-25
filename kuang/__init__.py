# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""kuang — a human-in-the-loop adversarial code-review panel.

A bounded, human-convened loop of reviewer personas — each tasked to *break* a
change, not approve it — with a ruin-class circuit-breaker. The panel informs; a
human adjudicates and decides.

Three concerns, three subpackages:

* ``kuang.engine``   — the deterministic loop: halting, dedup, budget, and the
                          UGLY circuit-breaker. Knows nothing about any runtime.
* ``kuang.backends`` — bindings to a particular agent runtime.
* ``kuang.cli``      — the entry point, the only place that knows both exist.

``python -m kuang`` runs it, as does the ``kuang`` command.

The name: ``blackice`` on PyPI is an unrelated project which ships a top-level
``blackice`` package *and* a ``blackice`` console script from the same entry point
this would have used. Installing both into one environment silently merges the two
directories, resolves the command to whichever landed last, and breaks the other's
uninstall. So this is ``kuang``, after the intrusion program that cuts through ICE —
which is what a panel tasked to break a change is for. ``blackice`` remains the name
of the repository it is developed in.
"""

__version__ = "0.1.0"
