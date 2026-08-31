# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Presentation: how a run's material is rendered for a human or a later run.

Rendering that more than one module needs lives here, so two callers cannot
drift apart. Nothing here decides anything — the panel is raw material, and the
synthesis is the human's.
"""

from __future__ import annotations


def ledger_line(*, severity: str, is_open: bool, persona: str, title: str,
                file: str | None, line: int | None,
                about_run: bool = False, ungrounded: bool = False) -> str:
    """Render one ledger entry: ``- [SEV/state] (persona) title @ file:line [tags]``.

    The line is a contract, not a convenience: ``session.on_epoch`` renders this
    run's ledger into cross-epoch memory, the CLI writes the same findings out as
    JSON, and ``memory.load_prior_findings`` renders them again on a later run
    seeded with ``--prior-findings``. The two renderings must agree byte-for-byte
    or a seeded re-run reads differently from the run that produced it — so there
    is one renderer, and ``tests/test_ledger_round_trip.py`` holds it to that.

    Takes primitives rather than a ``Finding`` because one caller has a Finding
    and the other has a JSON object; each keeps its own tolerance for absent
    fields, which is parsing, not presentation.

    The line carries per-finding **provenance** as well as per-finding **state**
    (#71). ``about_run`` says the finding is the instrument's own diagnosis
    (#73); ``ungrounded`` says the call that produced it opened nothing (#70).
    Both default to ``False`` so an artefact written before either existed — or by
    anything other than this CLI — renders exactly as it did, which is what keeps
    a cold seed byte-identical to the run that produced it. What each tag MEANS is
    stated where a persona reads it, in ``contract.build_prompt``: a tag nobody
    explained is decoration.
    """
    state = "open" if is_open else "resolved"
    loc = f"{file}:{line}" if file else "-"
    # Fixed order, so two renderings of one finding cannot differ by arrangement.
    tags = "".join(mark for flag, mark in
                   ((about_run, " [about the run]"), (ungrounded, " [ungrounded]"))
                   if flag)
    return f"- [{severity}/{state}] ({persona}) {title} @ {loc}{tags}"


def render_argv(argv: list[str]) -> str:
    """Render a ``claude`` argv for a human, eliding the two enormous arguments.

    Used by the dry run, whose only job is pre-flight confirmation: it reports
    the argv that would actually be spawned rather than a separately-written
    description of it, so the report cannot disagree with the call.
    """
    out: list[str] = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg in ("-p", "--append-system-prompt"):
            label = "prompt" if arg == "-p" else "system-prompt"
            out.append(f"{arg} <{label}: {len(argv[i + 1])} chars>")
            skip = True
        else:
            out.append(arg)
    return " ".join(out)
