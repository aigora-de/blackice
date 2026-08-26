# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The semantic reduce: an LLM clusterer for the ``Reduce`` seam (issue #1)."""

from __future__ import annotations

import json
from typing import Sequence

from kuang.engine import Cluster, Finding

from .contract import _collapse_whitespace, extract_json_blocks


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

# The example the contract shows, hoisted out of the prompt so the text we ship and
# the text ``_is_cluster_echo`` recognises are one string and cannot drift (#61).
# Unlike the findings template it is a *plausible real answer*, which is what made
# the echo dangerous: a clusterer restating it merged findings nobody grouped.
_CLUSTER_TEMPLATE = '{"clusters": [[0, 3], [1], [2]]}'

_CLUSTER_ECHO = _collapse_whitespace(_CLUSTER_TEMPLATE)

# Note the deliberate absence of a fenced ``json`` marker in the prose below: this
# contract used to name one inline, so a clusterer restating it handed the
# extraction regex an opening fence in the middle of a sentence and it captured a
# fragment of prose (#61, the twin of #26). The only literal fence here is the
# example's own.
_CLUSTER_CONTRACT = f"""
---
OUTPUT CONTRACT (mandatory). End your reply with EXACTLY one fenced `json` block
and nothing after it, assigning finding indices to groups. Every index from 0 to
N-1 must appear in exactly one group; singletons are expected and encouraged:

```json
{_CLUSTER_TEMPLATE}
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


def _is_cluster_echo(block: str) -> bool:
    """True if a payload is the contract's example restated rather than answered.

    Exact — whitespace-normalised equality with ``_CLUSTER_TEMPLATE`` and nothing
    else — for the same reason as ``contract._is_contract_echo``: this predicate
    causes a payload to be **discarded**, so the only text it may ever match is
    text we shipped.

    The tradeoff is sharper here than for the findings contract, and it is
    accepted rather than hidden. The findings template is full of obvious
    placeholders; this example is a *plausible real answer*, so a clusterer that
    genuinely decides exactly this grouping for exactly four findings is refused
    and its merge is lost. That failure runs in the safe direction — a duplicate
    survives — and the mandate this module ships says in as many words that
    merging distinct issues is worse than leaving duplicates. The undetected case
    runs the other way: findings merged because the *example* said so.
    """
    return _collapse_whitespace(block) == _CLUSTER_ECHO


def _extract_cluster_groups(text: str) -> list[list[int]] | None:
    """Pull ``[[0,3],[1],...]`` index-groups from a clusterer reply, or None.

    Tolerant: takes the last fenced ```json block that is not the contract's own
    example (or the whole text), accepts a ``clusters``/``groups`` key or a bare
    list, and normalises a stray bare int to a singleton group. Returns None when
    nothing list-shaped can be found, and when every candidate was the example
    echoed back — the caller then falls back to the identity reduce.

    The blocks themselves are found by ``contract.extract_json_blocks``: one
    contract, one parser. Which block to read, and falling back to the whole reply
    when there is no fence, are this caller's own tolerance — not the extractor's
    (#19, #26).
    """
    # No fence at all means read the whole reply, which is this caller's tolerance
    # and also the shape an *unfenced* echo arrives in, so it is filtered alike.
    candidates = extract_json_blocks(text) or [text]
    kept = [c for c in candidates if not _is_cluster_echo(c)]
    if not kept:
        return None                  # the contract restated; nothing was grouped
    raw = kept[-1]
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
