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
from typing import Iterable

from kuang.engine import Finding
from kuang.report import ledger_line


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
            f"{path}: no findings to seed. Expected a kuang run's JSON output "
            "(the object under '--- JSON ---', or its 'findings' array), or a raw "
            "run log containing that block.")
    lines = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        # Every field is read tolerantly (#25): an artefact carrying a "file" but
        # no "line" used to raise KeyError here and take the seeded re-run down at
        # startup — the one moment an operator has least context to read a
        # traceback. A missing line renders exactly as an absent one does on the
        # other side of the round trip, which is what keeps the two renderers
        # byte-identical.
        lines.append(ledger_line(
            severity=f.get("severity", "NOTE"), is_open=f.get("open", True),
            persona=f.get("persona", "?"), title=f.get("title", "(untitled)"),
            file=f.get("file"), line=f.get("line") if f.get("file") else None))
    if not lines:
        raise ValueError(f"{path}: findings present but none were readable.")
    return "\n".join(lines)


def epoch_summary(findings: Iterable[Finding]) -> str:
    """Render this run's ledger as cross-*epoch* memory for the next epoch.

    The counterpart of ``load_prior_findings``: same line, same renderer, so what
    a later run is seeded with reads exactly as what this run remembered.
    """
    return "\n".join(
        ledger_line(severity=f.severity.name, is_open=f.counts_open,
                    persona=f.persona, title=f.title, file=f.file, line=f.line)
        for f in findings)
