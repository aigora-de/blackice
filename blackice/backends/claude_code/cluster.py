# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The semantic reduce: an LLM clusterer for the ``Reduce`` seam (issue #1)."""

from __future__ import annotations

import json
import re
from typing import Sequence

from blackice.engine import Cluster, Finding


#
# This is the non-deterministic ``loop.Reduce`` implementation. The engine owns
# the deterministic identity default; here a cheap model folds re-worded /
# re-located dups of a single concept into one canonical cluster. Everything the
# engine's control logic depends on is DETERMINISTIC glue around the model call:
# the output is forced into a **partition** (every finding in exactly one cluster,
# never dropped), bad indices are sanitised, and any failure degrades to the
# identity reduce. Severity is UGLY-preserving by construction (``Cluster.severity``
# is the max of its members), so a merge can never hide a ruin-class finding.

_CLUSTER_MANDATE = (
    "You are a careful synthesiser. Group same-issue code-review findings "
    "conservatively. Do NOT review, add, drop, or re-severitise findings — only "
    "cluster the ones you are given.")

_CLUSTER_CONTRACT = """
---
OUTPUT CONTRACT (mandatory). End your reply with EXACTLY one fenced ```json block
and nothing after it, assigning finding indices to groups. Every index from 0 to
N-1 must appear in exactly one group; singletons are expected and encouraged:

```json
{"clusters": [[0, 3], [1], [2]]}
```
"""


def build_cluster_prompt(findings: Sequence[Finding]) -> str:
    """Assemble the clusterer task: a numbered finding list + the group contract."""
    lines = []
    for i, f in enumerate(findings):
        loc = f"{f.file}:{f.line}" if f.file else "-"
        lines.append(f"[{i}] ({f.severity.name}) [{f.claim_class}] {f.title} @ {loc}")
    return (
        "You are given a numbered list of code-review findings raised by several "
        "independent reviewers. Group ONLY those that describe the SAME underlying "
        "issue (same root cause / same defect) — even if worded differently, at "
        "different line numbers, or in different files.\n\n"
        "Be CONSERVATIVE: precision over recall. If in doubt, DO NOT merge — leave "
        "the finding in its own group. Merging distinct issues is worse than "
        "leaving duplicates.\n\n"
        f"FINDINGS:\n" + "\n".join(lines) + f"\n{_CLUSTER_CONTRACT}")


def _extract_cluster_groups(text: str) -> list[list[int]] | None:
    """Pull ``[[0,3],[1],...]`` index-groups from a clusterer reply, or None.

    Tolerant: takes the last fenced ```json block (or the whole text), accepts a
    ``clusters``/``groups`` key or a bare list, and normalises a stray bare int to
    a singleton group. Returns None only when nothing list-shaped can be found —
    the caller then falls back to the identity reduce.
    """
    blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    raw = blocks[-1] if blocks else text
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        groups = data.get("clusters", data.get("groups"))
    elif isinstance(data, list):
        groups = data
    else:
        groups = None
    if not isinstance(groups, list):
        return None
    out: list[list[int]] = []
    for g in groups:
        if isinstance(g, bool):        # bool is an int subclass — reject explicitly
            continue
        if isinstance(g, int):
            out.append([g])
        elif isinstance(g, list):
            out.append([i for i in g if isinstance(i, int) and not isinstance(i, bool)])
    return out


def _groups_to_clusters(findings: Sequence[Finding],
                        groups: list[list[int]]) -> list[Cluster]:
    """Turn model index-groups into a strict partition of ``findings``.

    Guarantees: each finding appears in exactly one cluster. Out-of-range and
    duplicate indices are ignored (first placement wins); any finding the model
    left unplaced becomes its own singleton — a panellist's claim is never dropped.
    """
    n = len(findings)
    assigned: set[int] = set()
    clusters: list[Cluster] = []
    for group in groups:
        members: list[Finding] = []
        for idx in group:
            if 0 <= idx < n and idx not in assigned:
                assigned.add(idx)
                members.append(findings[idx])
        if members:
            clusters.append(Cluster(members=tuple(members), title=members[0].title))
    for idx in range(n):               # unplaced findings -> singletons (never dropped)
        if idx not in assigned:
            f = findings[idx]
            clusters.append(Cluster(members=(f,), title=f.title))
    return clusters
