# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""CHARACTERISATION tests for the findings contract parser (``parse_findings``).

Not regression tests. They pin what the parser does **today**, bugs included, so
that the #19 restructure — which gives the fenced-JSON extraction exactly one
implementation shared with the clusterer — cannot change it unnoticed. The parser
had no direct coverage at all before this file, so three of #19's four
consolidations had no oracle.

Two of the behaviours pinned here are open defects, deliberately recorded as-is:

* an unrecognised severity silently becomes ``NOTE`` (#24 — a decorated ``UGLY``
  loses the circuit-breaker);
* a non-numeric ``line`` raises out of the parser (#25 — one persona's reply can
  kill a paid run).

When those are fixed in Epoch 2 these tests must be rewritten, not "updated".
"""

from __future__ import annotations

import pytest

from kuang.backends.claude_code.contract import parse_findings
from kuang.engine import Severity


def _reply(body: str) -> str:
    return f"Here is my review.\n\n```json\n{body}\n```\n"


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


# --- pinned defects (see the module docstring) ------------------------------

def test_an_unrecognised_severity_silently_becomes_note():
    """CHARACTERISATION of #24: a decorated UGLY loses the circuit-breaker."""
    report = parse_findings("p", _reply(
        '{"findings": [{"title": "t", "severity": "UGLY (ruin-class)"}]}'))

    assert report.findings[0].severity is Severity.NOTE


def test_a_severity_is_case_and_space_insensitive():
    report = parse_findings("p", _reply(
        '{"findings": [{"title": "t", "severity": "  blocker "}]}'))

    assert report.findings[0].severity is Severity.BLOCKER


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
