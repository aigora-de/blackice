# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Memory: what a panel is told about what was already found.

Two kinds, deliberately kept apart. Cross-*epoch* memory is this run's ledger,
rewritten each epoch. Cross-*run* memory is an earlier run's ledger loaded by
``--prior-findings`` (issue #13) and never rewritten — a re-run whose purpose is
confirming fixes gets it from epoch 1.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_prior_findings(path: str | Path) -> str:
    """Render a previous run's ledger as a seed summary (issue #13).

    Accepts, in order: a whole run envelope (``{"findings": [...]}`` — the object
    printed under ``--- JSON ---``), a bare findings array, or a raw run log with
    that block embedded.  The log is accepted because it is what the durable
    archive actually holds; making an operator carve the JSON out by hand is how a
    seeded re-run quietly gets nothing.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The file carries no recognisable findings.  Loud by design — a
            silently empty seed is indistinguishable from a working one, which is
            the failure mode issue #11 already recorded for the surface cap.
    """
    text = Path(path).read_text()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # A raw run log: take the LAST --- JSON --- block, which is the run's own
        # (an earlier one could be quoted inside a --why).
        blocks = text.split("--- JSON ---")
        if len(blocks) > 1:
            try:
                payload = json.loads(blocks[-1].strip())
            except json.JSONDecodeError:
                payload = None
    findings = payload.get("findings") if isinstance(payload, dict) else payload
    if not isinstance(findings, list) or not findings:
        raise ValueError(
            f"{path}: no findings to seed. Expected a blackice run's JSON output "
            "(the object under '--- JSON ---', or its 'findings' array), or a raw "
            "run log containing that block.")
    lines = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        state = "open" if f.get("open", True) else "resolved"
        loc = f"{f['file']}:{f['line']}" if f.get("file") else "-"
        lines.append(f"- [{f.get('severity', 'NOTE')}/{state}] ({f.get('persona', '?')}) "
                     f"{f.get('title', '(untitled)')} @ {loc}")
    if not lines:
        raise ValueError(f"{path}: findings present but none were readable.")
    return "\n".join(lines)
