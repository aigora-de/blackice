# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Memory: what a panel is told about what was already found.

Two kinds, deliberately kept apart. Cross-*epoch* memory is this run's ledger,
rewritten each epoch. Cross-*run* memory is an earlier run's ledger loaded by
``--prior-findings`` (issue #13) and never rewritten — a re-run whose purpose is
confirming fixes gets it from epoch 1.

Both are rendered by one line, and that line carries **provenance** (#71): the
memory a panel is given says which of its lines this run knows were not grounded
in the source. See ``epoch_summary`` for the rule and what it deliberately leaves
to a sibling issue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from kuang.engine import Finding, PersonaReport, ReviewRun
from kuang.report import ledger_line

from .spawn import called_no_tool


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
            file=f.get("file"), line=f.get("line") if f.get("file") else None,
            about_run=bool(f.get("about_run")),
            ungrounded=bool(f.get("ungrounded"))))
    if not lines:
        raise ValueError(f"{path}: findings present but none were readable.")
    return "\n".join(lines)


def ungrounded_keys(run: ReviewRun) -> frozenset[str]:
    """The ledger keys whose finding was first raised by a call that opened nothing.

    The join #71 needs, and the reason it is a join: a ``Finding`` carries no link
    back to the call that produced it, while the degradations are facts about one
    spawn and live on ``PersonaReport``. ``on_epoch`` receives the whole
    ``ReviewRun``, so ``run.epochs[*].reports`` is in scope and the engine's seam
    does not have to widen for this.

    Keyed on **(persona, epoch)** and not on ``f.persona``: a persona healthy in
    epoch 1 and degraded in epoch 2 is exactly the case a persona-only key gets
    wrong. The ledger stores **first sighting** by signature, so this walks the
    epochs in order and takes the first report to raise each key — mirroring
    ``loop.run``'s own insertion, ``counts_open`` filter included, so the two
    cannot disagree about which call a ledger entry came from.

    The predicate is ``spawn.called_no_tool`` under ``status.reviewed``, reused
    rather than restated — it is the same conjunction the participation line
    prints (``cli._turn_note``). A call that errored, crashed or never happened has
    no review to be ungrounded, and saying so twice under two headings is what
    that guard exists to prevent.
    """
    first: dict[str, PersonaReport] = {}
    for epoch in run.epochs:
        for report in epoch.reports:
            for f in report.findings:
                if f.counts_open:
                    first.setdefault(f.key, report)
    return frozenset(key for key, report in first.items()
                     if report.status.reviewed and called_no_tool(report.turns))


def epoch_summary(findings: Iterable[Finding],
                  ungrounded: frozenset[str] = frozenset()) -> str:
    """Render this run's ledger as cross-*epoch* memory for the next epoch.

    The counterpart of ``load_prior_findings``: same line, same renderer, so what
    a later run is seeded with reads exactly as what this run remembered.

    **The rule (#71): a line this run knows was not grounded in the source is
    marked, and nothing is excluded.** ``on_epoch`` rewrites the next epoch's
    prior context from the whole ledger, so without this a finding a degraded call
    produced is handed to personas that did not degrade — and it compounds, since
    epoch 3 is shaped by epochs 1 and 2 together and ``--prior-findings`` carries
    the result into a later run where nothing re-adjudicates it against source.

    Two facts establish "not grounded" exactly, and both are **set at the source**:

    * ``Finding.about_run`` (#73) — the finding is the instrument's own diagnosis
      and not a claim about the source at all;
    * ``ungrounded`` — the key is in the set ``ungrounded_keys`` returns, i.e.
      the call that first raised it called no tool (#70).

    Marked, never filtered: #71's acceptance says *unmarked*, not *excluded*, and
    an exclusion here would make a failed persona's diagnosis unrecoverable from
    the context the next panel reads. Neither fact is a predicate over
    model-produced text, which is the constraint #73 **measured**: excluding
    ``claim_class == "meta"`` turns a run that halts ``escalate_ugly`` on a
    persona-declared ``meta`` UGLY into one that halts ``converged`` with none, and
    the same bars a match on a finding's title (#30 refused that one layer up).

    What it deliberately does **not** cover, each with the sibling that owns it: a
    **reduced surface** (#69) — the reviewer looked, at less, which is a fact about
    what it saw rather than whether it looked, and is per epoch rather than per
    call; a **refused tool** (#67) — a reviewer denied ``Bash`` may still have read
    every file, so groundedness is not knowable from a refusal, which is why the
    run states a refusal and declines to classify it; and a finding **fabricated**
    by the reformat retry (#63) — it carries whatever the model wrote and no flag
    our code sets can name it. A finding first raised by a healthy call and
    re-raised by a degraded one reads healthy, which is the ledger's own
    first-sighting identity rule and not this one's.
    """
    return "\n".join(
        ledger_line(severity=f.severity.name, is_open=f.counts_open,
                    persona=f.persona, title=f.title, file=f.file, line=f.line,
                    about_run=f.about_run, ungrounded=f.key in ungrounded)
        for f in findings)
