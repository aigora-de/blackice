# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Tests for the findings contract parser (``parse_findings``).

The file began as a CHARACTERISATION set written during #19: it pinned what the
parser did **then**, bugs included, so that the restructure — which gives the
fenced-JSON extraction exactly one implementation shared with the clusterer —
could not change it unnoticed. The parser had no direct coverage at all before it,
so three of #19's four consolidations had no oracle.

One of the two defects it pinned is now fixed, and its characterisation test was
rewritten rather than updated:

* an unresolvable severity no longer silently becomes ``NOTE`` (#24). A decorated
  ``UGLY`` keeps the circuit-breaker; a value the vocabulary cannot resolve is
  escalated to ``UNRESOLVED_SEVERITY`` and **recorded**, never quietly downgraded.
  Those are regression tests: each goes red without the fix.

One characterisation test remains, pinning a defect that is still open:

* a non-numeric ``line`` raises out of the parser (#25 — one persona's reply can
  kill a paid run). It is that issue's oracle; leave it as it is until #25.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code.contract import (FINDINGS_CONTRACT,
                                                 UNRESOLVED_SEVERITY,
                                                 parse_findings)
from kuang.engine import Severity


def _reply(body: str) -> str:
    return f"Here is my review.\n\n```json\n{body}\n```\n"


def _one_finding(**fields) -> str:
    """A reply carrying exactly one finding, built through ``json`` so that a
    severity containing quotes, pipes or braces cannot break the fixture."""
    return _reply(json.dumps({"findings": [{"title": "t", **fields}]}))


# --- the happy path ---------------------------------------------------------

def test_a_well_formed_reply_yields_verdict_and_findings():
    report = parse_findings("engineer", _reply(
        '{"verdict": "NO", "findings": [{"title": "off-by-one", '
        '"severity": "BLOCKER", "claim_class": "alloc", "file": "x.py", '
        '"line": 42, "evidence": "read it"}]}'))

    assert report.persona == "engineer"
    assert report.verdict == "NO"
    assert len(report.findings) == 1
    f = report.findings[0]
    assert (f.title, f.severity, f.claim_class) == ("off-by-one", Severity.BLOCKER, "alloc")
    assert (f.file, f.line, f.evidence) == ("x.py", 42, "read it")


def test_the_last_fenced_block_wins():
    """The contract says the JSON block ends the reply; earlier blocks are quoted."""
    text = _reply('{"verdict": "YES", "findings": []}') + _reply(
        '{"verdict": "NO", "findings": [{"title": "real", "severity": "UGLY", '
        '"claim_class": "ruin"}]}')

    report = parse_findings("p", text)

    assert report.verdict == "NO"
    assert [f.title for f in report.findings] == ["real"]


def test_missing_fields_take_their_defaults():
    report = parse_findings("p", _reply('{"findings": [{}]}'))

    f = report.findings[0]
    assert report.verdict is None
    assert (f.title, f.severity, f.claim_class) == ("", Severity.NOTE, "uncategorised")
    assert (f.file, f.line, f.evidence) == (None, None, "")
    assert report.unresolved_severities == []


@pytest.mark.parametrize("raw, expected", [
    (0, None),            # a 0 line is indistinguishable from "no line"
    (None, None),
    ("", None),
    ("17", 17),           # a numeric string is coerced
])
def test_line_coercion(raw, expected):
    report = parse_findings("p", _reply(
        '{"findings": [{"title": "t", "line": %s}]}' % ("null" if raw is None
                                                        else repr(raw).replace("'", '"'))))

    assert report.findings[0].line == expected


# --- the contract-miss sentinel ---------------------------------------------

def test_no_fenced_block_yields_the_meta_sentinel():
    """The shape ``_is_parse_failure`` (and so the retry path) keys on."""
    report = parse_findings("p", "I reviewed it and found nothing to say.")

    assert report.verdict is None
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.claim_class == "meta"
    assert f.title == "no structured output (parse failure)"
    assert f.severity is Severity.NOTE
    assert f.evidence == "I reviewed it and found nothing to say."


def test_the_sentinel_evidence_is_capped_at_400_characters():
    report = parse_findings("p", "x" * 1000)

    assert len(report.findings[0].evidence) == 400


def test_unparseable_json_yields_a_distinct_sentinel():
    report = parse_findings("p", _reply("{not json"))

    f = report.findings[0]
    assert f.claim_class == "meta"
    assert f.title.startswith("unparseable JSON findings:")
    assert f.evidence == "{not json\n"


# --- severity: the leading label, and nothing guessed (#24) -----------------

def test_a_decorated_ugly_keeps_the_circuit_breaker():
    """REGRESSION for #24: this landed as NOTE, three rungs below the breaker."""
    report = parse_findings("p", _one_finding(severity="UGLY (ruin-class)"))

    assert report.findings[0].severity is Severity.UGLY
    assert report.unresolved_severities == []


@pytest.mark.parametrize("raw, expected", [
    ("UGLY", Severity.UGLY),
    ("NOTE", Severity.NOTE),
    ("  blocker ", Severity.BLOCKER),             # case- and space-insensitive
    ("UGLY (ruin-class)", Severity.UGLY),         # the decoration #24 reports
    ("ugly - data loss", Severity.UGLY),
    ("BLOCKER (not UGLY)", Severity.BLOCKER),     # a negated mention is prose
    ("non-blocking", Severity.NON_BLOCKING),      # our own word, separated
    ("Non Blocking", Severity.NON_BLOCKING),
    ("NON_BLOCKING", Severity.NON_BLOCKING),
])
def test_the_leading_label_is_the_severity(raw, expected):
    """The label leads; whatever follows it is prose we do not map."""
    report = parse_findings("p", _one_finding(severity=raw))

    assert report.findings[0].severity is expected
    assert report.unresolved_severities == []


@pytest.mark.parametrize("raw", [
    "CRITICAL", "HIGH", "P1",                     # words our vocabulary lacks
    "", "   ",                                    # present but empty
    "not a blocker", "non blocker",               # prose, not a label
    "a critical blocker",                         # a label, but not leading
    "Blocker/Ugly", "UGLY/BLOCKER",               # a declared tie: two levels, no choice
    "NOTE | NON_BLOCKING | BLOCKER | UGLY",       # the contract template, echoed
])
def test_an_unresolved_severity_is_escalated_and_recorded(raw):
    """Both halves matter: the level it takes AND the fact that we said so.

    A test asserting only the severity would pass a parser that defaults to
    BLOCKER in silence — which is #24 again, one rung up.
    """
    report = parse_findings("p", _one_finding(severity=raw))

    assert report.findings[0].severity is UNRESOLVED_SEVERITY
    assert report.unresolved_severities == [raw]


def test_the_unresolved_level_is_visible_to_the_gate():
    """Decision 1 as a property, not as a constant.

    Asserting ``severity is UNRESOLVED_SEVERITY`` elsewhere is tautological — flip
    the constant and those tests follow it. What must hold is that the level an
    escalated severity takes is one the gate can see: below BLOCKER it cannot
    block convergence (``_evaluate_halt``), so a value nobody could read would be
    swallowed after all, which is #24 wearing a different number.
    """
    assert UNRESOLVED_SEVERITY >= Severity.BLOCKER


@pytest.mark.parametrize("raw", ["CRITICAL", "", "not a blocker", "Blocker/Ugly"])
def test_an_unresolved_severity_is_never_silently_noted(raw):
    """#24's sentence, executable: never a quiet NOTE."""
    report = parse_findings("p", _one_finding(severity=raw))

    assert not (report.findings[0].severity is Severity.NOTE
                and not report.unresolved_severities)


def test_a_non_string_severity_is_unresolved_not_a_crash():
    """The payload is model-produced: a number must escalate, not raise."""
    report = parse_findings("p", _one_finding(severity=3))

    assert report.findings[0].severity is UNRESOLVED_SEVERITY
    assert report.unresolved_severities == ["3"]


def test_a_recorded_raw_severity_is_capped():
    """A model-controlled string reaches the run artefact; bound it there."""
    report = parse_findings("p", _one_finding(severity="Z" * 500))

    assert report.unresolved_severities == ["Z" * 120]


def test_an_absent_severity_keeps_its_default_and_records_nothing():
    """The boundary: a finding that claimed no severity is not one we misread."""
    report = parse_findings("p", _one_finding())

    assert report.findings[0].severity is Severity.NOTE
    assert report.unresolved_severities == []


def test_every_persona_records_its_own_unresolved_severities():
    report = parse_findings("p", _reply(json.dumps({"findings": [
        {"title": "a", "severity": "CRITICAL"},
        {"title": "b", "severity": "UGLY"},
        {"title": "c", "severity": "HIGH"}]})))

    assert [f.severity for f in report.findings] == [
        UNRESOLVED_SEVERITY, Severity.UGLY, UNRESOLVED_SEVERITY]
    assert report.unresolved_severities == ["CRITICAL", "HIGH"]


# --- the contract asks for what the parser reads ----------------------------

def test_the_contract_defines_the_whole_vocabulary():
    """Every level defined, not just the two that were (#24 decision 4)."""
    for sev in Severity:
        # Twice: once in the JSON template, once where the levels are defined.
        assert FINDINGS_CONTRACT.count(sev.name) >= 2, sev.name


def test_the_contract_sanctions_an_undecided_persona():
    """A reviewer that cannot choose must not be made to invent certainty."""
    assert "BLOCKER/UGLY" in FINDINGS_CONTRACT


# --- pinned defect: still open (see the module docstring) -------------------

def test_a_non_numeric_line_raises_out_of_the_parser():
    """CHARACTERISATION of #25: this exception kills the whole run."""
    with pytest.raises(ValueError):
        parse_findings("p", _reply('{"findings": [{"title": "t", "line": "~120"}]}'))


# --- one extractor for one contract (#19) -----------------------------------

def test_the_extractor_returns_the_last_fenced_block():
    from kuang.backends.claude_code.contract import extract_json_block

    assert extract_json_block(_reply("A") + _reply("B")) == "B\n"


def test_the_extractor_returns_none_when_there_is_no_fence():
    from kuang.backends.claude_code.contract import extract_json_block

    assert extract_json_block("no block here") is None


def test_the_clusterer_uses_the_same_extractor():
    """Two parsers for one contract meant a fix to one silently left the other."""
    from kuang.backends.claude_code import cluster, contract

    assert cluster.extract_json_block is contract.extract_json_block
