# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The deterministic default ``Reduce``: identity over the dedup signature.

The engine owns this one because the engine's control logic depends on it; a
semantic (LLM) reducer is a backend concern and degrades to exactly this when it
is unavailable. See ``blackice.backends.claude_code.cluster``.
"""

from __future__ import annotations

from typing import Sequence

from .findings import Cluster, Finding


def _identity_reduce(findings: Sequence[Finding]) -> list[Cluster]:
    """Deterministic default reduce: one cluster per distinct signature.

    Groups by ``Finding.key`` — reproducing today's signature dedup exactly. Each
    finding becomes its own singleton cluster (``cluster.key == finding.key``), so
    the cluster-level control logic collapses to the historical key-based
    behaviour whenever no semantic reducer is injected.
    """
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault(f.key, []).append(f)
    return [Cluster(members=tuple(v), title=v[0].title) for v in groups.values()]
