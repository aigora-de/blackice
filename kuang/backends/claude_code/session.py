# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The session: the engine's seams, bound to the ``claude`` CLI.

``PanelSession`` is wiring. It holds the run's cross-epoch state and hands each
seam to the module that owns that job — ``surface`` gathers, ``spawn`` calls out,
``contract`` assembles and parses, ``cluster`` reduces, ``memory`` remembers.

Binds ``kuang.engine.SpawnPersona`` to the ``claude`` binary in headless print
mode: **one ``claude -p`` subprocess per persona per epoch**, driven from the
terminal. No SDK dependency — it shells out to the same ``claude`` the user runs,
so it "works in Claude Code" by *being* Claude Code.

Design choices realised across this package (see
``two-pass-adversarial-review-pattern.md`` and the design discussion):

* **Persona sourcing (layered):** ``personas`` — parse ``CLAUDE.md`` "Resident
  Experts" when present -> else ``panel.yaml`` / ``panel.md`` -> else a distilled
  default set.
* **Open-ended mandates.** A persona's *identity* (its role/responsibilities from
  ``CLAUDE.md``) is its lens; we do not impose a prescriptive checklist that would
  "lead the witness".
* **Tools as behavioural grounding.** Every reviewer gets read-only source
  inspection (``Read``/``Grep``/``Glob``) and **no edit tools** — biasing them to
  verify against source rather than speculate. Test/lint *execution* is **not** the
  default and never has been: ``Bash`` is deny-listed, so a reviewer cannot run
  ``pytest``, ``git`` or ``ruff`` at all unless an operator opts in with a scoped
  ``--allow-tools 'Bash(pytest:*)'`` — and that grant does **not** bound what is then
  run: measured, under the default ``plan`` mode it is an unrestricted shell and
  nothing records it (#96). The deny-list likewise bounds the tools it NAMES, not the
  whole tool set (#76). The policy lives in ``permissions``.
* **Independent within an epoch;** epoch > 1 is handed *all* prior epochs'
  findings (cross-epoch memory), so the panel builds on itself without
  intra-epoch debate (which would reintroduce groupthink).
* **Structured findings** enforced by an output contract appended to each prompt.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from kuang.engine import (Cluster, EpochResult, Finding, GateDecision,
                             PersonaReport, PersonaStatus, ReviewRun, ReviewSpec,
                             Severity)
from kuang.engine.reduce import _identity_reduce
from kuang.report import render_argv

from .cluster import (_CLUSTER_MANDATE, ReduceState, _extract_cluster_groups,
                      _groups_to_clusters, build_cluster_prompt)
from .contract import (FINDINGS_CONTRACT, UNRESOLVED_SEVERITY,
                       _is_parse_failure, build_prompt, parse_findings)
from .memory import epoch_summary, ungrounded_keys
from .permissions import DEFAULT_DISALLOWED_TOOLS
from .personas import Persona
from .spawn import CallResult, _resolve_claude_bin, build_argv, run_claude
from .surface import SurfaceRecord, build_path_surface, gather_diff


@dataclass
class PanelSession:
    """Wires the loop seams to the ``claude`` CLI and carries cross-epoch memory."""

    repo_root: Path
    spec: ReviewSpec
    personas: dict[str, Persona]
    base: str
    head: str = "HEAD"
    paths: list[str] | None = None      # path mode (issue #6): review these files/dirs
    max_surface_bytes: int = 200_000    # total-size cap for the path-mode surface
    default_model: str | None = None
    cluster_model: str | None = None    # model for the semantic reduce (cheap; None -> default)
    dry_run: bool = False
    claude_bin: str = field(default_factory=_resolve_claude_bin)
    disallowed_tools: list[str] = field(
        default_factory=lambda: list(DEFAULT_DISALLOWED_TOOLS))
    permission_mode: str = "plan"
    tokens: int = 0
    # One state per call to ``reduce``, i.e. one per epoch, in order (#30). Kept
    # HERE, on the backend, rather than returned through the engine's ``Reduce``
    # seam: that seam returns ``list[Cluster]`` and nothing else, the engine never
    # reads this and never branches on it, and widening it would oblige every
    # implementation — the deterministic identity default included — to produce a
    # diagnostic about a clusterer it does not have. ``tokens`` above is the same
    # kind of fact about the same calls and lives in the same place.
    reduce_states: list[ReduceState] = field(default_factory=list)
    # One record per call to ``gather``, i.e. one per epoch, in order (#69), and
    # kept here for the reason ``reduce_states`` is: the engine's ``GatherSurface``
    # seam returns the surface text and nothing else, the engine never reads this
    # and never branches on it, and widening the seam would oblige every
    # implementation — a backend with no cap included — to produce a diagnostic
    # about one. A surface is gathered ONCE PER EPOCH and handed to every persona
    # unchanged, so this is an epoch fact and not a spawn fact: putting it on
    # seven ``PersonaReport``s would repeat one fact seven times and imply the
    # personas could differ, which they cannot.
    surface_states: list[SurfaceRecord] = field(default_factory=list)
    prior_summary: str = ""      # cross-EPOCH memory; on_epoch REWRITES it each epoch
    # Cross-RUN memory (issue #13): set once at startup from --prior-findings and
    # never touched again. Deliberately NOT folded into prior_summary, which
    # on_epoch overwrites from the current ledger — a seed stored there would be
    # discarded after epoch 1, losing exactly the memory the run was asked to
    # carry. Kept separate all the way to build_prompt, which labels the two so a
    # persona can tell "found before the fixes" from "found this run".
    seeded_summary: str = ""

    # --- gather: the review surface (re-read each epoch so fixes are visible) ---
    def gather(self, epoch: int) -> str:  # noqa: ARG002
        """Assemble the epoch's review surface, or refuse to run.

        A surface that could not be built is an operator error, never a review
        with nothing in it: git's return code and stderr are both checked, so a
        mistyped ref stops the run instead of handing the panel an empty string
        to unanimously approve.

        What each epoch's surface cost is recorded in ``surface_states`` (#69).
        A run that refuses records nothing: it never reached a panel, so there is
        no reduced surface to report and none is invented.

        Raises:
            SurfaceError: git failed, or the requested surface is empty.
        """
        # One shape, both modes (#74): each assembler returns the surface and a
        # record of what it was made of. Diff mode's used to be a module constant,
        # which could carry neither its refs nor its size.
        surface, record = (
            # path mode: full content of the named files/dirs (issue #6)
            build_path_surface(self.repo_root, self.paths, self.max_surface_bytes)
            if self.paths else
            gather_diff(self.repo_root, self.base, self.head))
        self.surface_states.append(record)
        return surface

    # --- the argv this session would spawn: the call and the dry run agree ---
    def _argv(self, prompt: str, mandate: str, tools: list[str],
              model: str | None) -> list[str]:
        return build_argv(
            claude_bin=self.claude_bin, prompt=prompt, mandate=mandate,
            tools=tools, disallowed_tools=self.disallowed_tools,
            permission_mode=self.permission_mode, repo_root=self.repo_root,
            model=model)

    # --- one `claude -p` call: returns what the envelope said about it ---
    def _run_claude(self, prompt: str, mandate: str, tools: list[str],
                    model: str | None) -> CallResult:
        return run_claude(self._argv(prompt, mandate, tools, model),
                          cwd=self.repo_root)

    # --- spawn: one persona review, with retry-on-contract-miss ---
    def spawn(self, persona: str, mandate: str, surface: str, epoch: int) -> PersonaReport:
        p = self.personas[persona]
        model = p.model or self.default_model
        surface_kind = "files" if self.paths else "diff"
        prompt = build_prompt(self.spec, surface, epoch, self.prior_summary,
                              surface_kind, seeded=self.seeded_summary)

        if self.dry_run:
            # Report the argv that would actually be spawned, not a separate
            # description of it: pre-flight confirmation is the dry run's only
            # job, so it must not be able to disagree with the call.
            preview = (prompt[:280] + "…") if len(prompt) > 280 else prompt
            print(f"\n[dry-run] persona={persona} model={model or 'default'}"
                  f"\n  argv= {render_argv(self._argv(prompt, mandate, p.tools, model))}"
                  f"\n  prompt≈ {preview!r}")
            # Says so, rather than passing for a persona that reviewed and found
            # nothing (#30) — which is exactly what an absent status let it do.
            # And casts NO VOTE (#72): this branch used to return a fabricated
            # "YES", so the whole panel met unanimity on calls nobody made and the
            # run halted `converged` — a good verdict for a review that never
            # happened. A verdict is a claim about a review; there is no review
            # here, so there is nothing to claim. Our own code must not invent a
            # control value for a call it did not make, which is the same rule
            # #67, #69, #70 and #71 each applied to a measurement not taken.
            return PersonaReport(persona=persona, verdict=None,
                                 status=PersonaStatus.NOT_SPAWNED)

        call = self._run_claude(prompt, mandate, p.tools, model)
        self.tokens += call.output_tokens
        # Tools this persona asked for and was refused (#67). Accumulated across
        # BOTH calls this method can make: the reformat retry is a second call and
        # can be starved too, and a report naming only one of them would understate
        # what the reviewer went without. Set at the source, never derived.
        denied = list(call.denied_tools)
        # What the review cost in turns (#70). The REVIEW call's count, and only
        # it: the reformat retry below is a formatter that needs no tool, and the
        # only ONE-turn call in this issue's whole measurement was one. Summing
        # the two would render 1 + 5 as 6 and launder an ungrounded review into a
        # grounded-looking one; taking the later count would report the
        # formatter's turns for a review that opened nothing, or flag a healthy
        # seven-turn review that merely missed the contract. The review is the
        # thing that was grounded or was not.
        turns = call.num_turns
        if call.error:
            # One channel, one wording (#29) — and now a status set beside it, so
            # #30 does not have to recover "this persona failed" by matching
            # ``agent error:`` off a finding's title.
            return PersonaReport(persona=persona, verdict=None,
                                 status=PersonaStatus.AGENT_ERROR,
                                 denied_tools=denied, turns=turns, findings=[
                Finding(persona, call.error, Severity.NOTE, "meta",
                        about_run=True)])
        report = parse_findings(persona, call.text)
        report.tokens = call.output_tokens
        report.denied_tools = list(denied)
        report.turns = turns

        # Retry-on-contract-miss: the persona reviewed but did not emit the JSON
        # contract (so its findings were lost). Reformat its raw review into the
        # contract via one cheap follow-up call rather than discarding it.
        if _is_parse_failure(report):
            reformat = (
                "Extract the findings from the review below into the EXACT JSON "
                "contract. Output ONLY the fenced ```json block, nothing else.\n\n"
                f"REVIEW:\n{call.text}\n\n{FINDINGS_CONTRACT}")
            retry = self._run_claude(
                reformat, "You are a formatter: reformat, do not review.", ["Read"], model)
            self.tokens += retry.output_tokens
            for tool in retry.denied_tools:
                if tool not in denied:
                    denied.append(tool)
            report.denied_tools = list(denied)
            if not retry.error:
                report2 = parse_findings(persona, retry.text)
                if not _is_parse_failure(report2):
                    report2.tokens = call.output_tokens + retry.output_tokens
                    report2.denied_tools = list(denied)
                    report2.turns = turns
                    return report2
        return report

    # --- reduce: the semantic clusterer (loop.Reduce), degrades to identity ---
    def reduce(self, findings: Sequence[Finding]) -> list[Cluster]:
        """Fold the deduped ledger into canonical clusters via one cheap model call.

        A ``loop.Reduce`` implementation. Never raises: a call error, a missing
        contract, or fewer than two findings all fall back to the deterministic
        identity reduce, so the engine's control loop keeps working even when the
        clusterer is unavailable.

        Every one of those outcomes is now RECORDED on the session, in ``reduce_states``,
        one entry per epoch (#30). Falling back silently is what made a dead
        clusterer and a live one that merged nothing byte-identical in a run's own
        output. What the engine gets back is unchanged — a partition of the input.

        The dry run is tested before the finding count so the reason reported is
        the truer one: in a dry run nothing was spawned, so there was never
        anything to reduce. Both return the identity reduce either way.
        """
        findings = list(findings)
        if self.dry_run:
            self.reduce_states.append(ReduceState.DRY_RUN)
            return _identity_reduce(findings)
        if len(findings) < 2:
            self.reduce_states.append(ReduceState.NOTHING_TO_REDUCE)
            return _identity_reduce(findings)
        model = self.cluster_model or self.default_model
        call = self._run_claude(
            build_cluster_prompt(findings), _CLUSTER_MANDATE, ["Read"], model)
        self.tokens += call.output_tokens
        if call.error:
            self.reduce_states.append(ReduceState.CALL_FAILED)
            return _identity_reduce(findings)
        groups, state = _extract_cluster_groups(call.text)
        self.reduce_states.append(state)
        if groups is None:
            return _identity_reduce(findings)
        return _groups_to_clusters(findings, groups)

    # --- checkpoint: refresh cross-epoch memory from the ledger ---
    def on_epoch(self, run: ReviewRun) -> None:
        """Rewrite the next epoch's prior context from the whole ledger.

        The whole ledger, still: nothing is excluded (#71). What is added is the
        provenance the next panel needs to weigh a line — which of them this run
        knows were not grounded in the source. The join is available here and
        nowhere downstream, because ``epoch_summary`` sees findings while the
        degradations are facts about a call; ``run`` carries both.
        """
        self.prior_summary = epoch_summary(run.ledger.values(),
                                           ungrounded_keys(run))

    def budget_spent(self) -> int:
        return self.tokens

    # --- gate: the HITL touchpoint between epochs ---
    def interactive_gate(self, result: EpochResult, run: ReviewRun) -> GateDecision:
        print(f"\n=== epoch {result.index} synthesis ===")
        print(f"new findings: {len(result.new_findings)} | open blockers: "
              f"{result.open_blockers} | open uglies: {result.open_uglies} | "
              f"tokens: {self.tokens}")
        for f in result.new_findings:
            print(f"  [{f.severity.name}] ({f.persona}) {f.title}")
        # A severity we could not read is reported here as well as at the end of
        # the run: the gate is where a human decides whether to keep spending, and
        # they cannot weigh a finding whose level the panel never actually set.
        for report in result.reports:
            for raw in report.unresolved_severities:
                print(f"  [unresolved severity] ({report.persona}) {raw!r} — "
                      f"escalated to {UNRESOLVED_SEVERITY.name}")
            # Likewise a verdict that was not a vote (#26): the gate is where a
            # human decides whether to keep spending, and a panel that looks
            # short of quorum for a reason nobody can see is a run that lies.
            if report.unresolved_verdict is not None:
                print(f"  [unresolved verdict] ({report.persona}) "
                      f"{report.unresolved_verdict!r} — not counted as a vote")
        if not sys.stdin.isatty():
            return GateDecision(stop=False)
        ans = input("gate — [c]ontinue / [s]top? ").strip().lower()
        return GateDecision(stop=ans.startswith("s"))
