# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Tests for the findings contract parser (``parse_findings``).

Regression tests. The file began as a CHARACTERISATION set written during #19: it
pinned what the parser did **then**, bugs included, so that the restructure —
which gives the fenced-JSON extraction exactly one implementation shared with the
clusterer — could not change it unnoticed. The parser had no direct coverage at
all before it, so three of #19's four consolidations had no oracle.

Both defects it pinned are now fixed — their characterisation tests were
rewritten rather than updated — and a third has since been fixed here:

* an unresolvable severity no longer silently becomes ``NOTE`` (#24). A decorated
  ``UGLY`` keeps the circuit-breaker; a value the vocabulary cannot resolve is
  escalated to ``UNRESOLVED_SEVERITY`` and **recorded**, never quietly downgraded.
* a model-shaped ``line`` no longer raises out of the parser (#25). Nothing in a
  contract payload can raise now: the whole thing is untrusted input, coerced or
  reported, because one persona's reply used to be able to kill a paid run.
* an echoed output contract no longer discards the review it followed, nor votes
  (#26). Block selection skips a block that is verbatim our own template, and the
  verdict is categorical: one word, ``YES`` or ``NO``, or it is not a vote. This
  one had no characterisation test — it was found by reading, not by the suite.
* a persona's status is set by the parser rather than inferred from the shape of
  its report (#30). ``_is_parse_failure`` used to read "no verdict, one finding,
  claim_class == 'meta'" as *unreadable*, which a genuine review can wear; one
  characterisation test pinned that false retry as benign and was converted.

So nothing here characterises a defect any more; each test goes red without its
fix. The contract payload is model-produced, which is why so many of these read
like abuse — they are the shapes a reasonable reviewer actually emits.
"""

from __future__ import annotations

import json

import pytest

from kuang.backends.claude_code.contract import (FINDINGS_CONTRACT,
                                                 UNRESOLVED_SEVERITY,
                                                 _TEMPLATE_BLOCK,
                                                 _is_parse_failure,
                                                 parse_findings)
from kuang.engine import PersonaStatus, Severity


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
    # --- #25: model-shaped locations. Each of the first five raised. ---
    ("~120", 120),        # an approximate location
    ("42-58", 42),        # a range: a human looks at the start of it
    ("L42", 42),          # a prefixed location
    ("line 42", 42),
    ("12.7", 12),         # a float as a string (a real float already truncated)
    (12.7, 12),
    ([1, 2], None),       # a container names no single line
    ({"a": 1}, None),
    (True, None),         # bool is an int subclass — never line 1
    ("no idea", None),
    ("0", None),          # "0" and 0 now agree
    (-5, None),           # a line is a positive integer or it is absent
    ("-5", 5),            # ...and the minus is punctuation, not a sign
])
def test_line_coercion(raw, expected):
    """A location is advisory, so it is read best-effort — but never fatally.

    Best-effort here and refusal-to-guess for severity (#24) are the same
    doctrine, not opposite ones: severity drives the breaker and the convergence
    gate, a location drives nothing. ``Finding.key`` buckets it by ``// 10``
    precisely because it is approximate.
    """
    report = parse_findings("p", _one_finding(line=raw))

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


# --- the payload is untrusted input, everywhere (#25) -----------------------

def test_a_model_shaped_line_does_not_kill_the_run():
    """REGRESSION for #25: this raised, out of a worker thread, killing the run."""
    report = parse_findings("p", _one_finding(line="~120", title="approximate"))

    assert report.findings[0].line == 120
    assert report.findings[0].title == "approximate"


@pytest.mark.parametrize("payload", [
    '{"findings": "none"}',                 # a string
    '{"findings": {"a": 1}}',               # an object
    '{"findings": 3}',                      # a number
    '{"findings": true}',
    '[{"title": "t"}]',                     # the payload itself is a list
    '"all good"',                           # ...or a string
    '42',                                   # ...or a number
])
def test_an_unusable_payload_becomes_the_contract_miss_sentinel(payload):
    """Unreachable findings take the path that already exists — and its retry.

    ``_is_parse_failure`` keys on this shape, so ``session.spawn`` reformats the
    reply into the contract instead of discarding the review.
    """
    report = parse_findings("p", _reply(payload))

    assert _is_parse_failure(report)
    assert report.verdict is None
    assert report.findings[0].claim_class == "meta"
    assert "contract violated" in report.findings[0].title


@pytest.mark.parametrize("payload", [
    '{"verdict": "YES"}',                   # reviewed, found nothing
    '{"verdict": "YES", "findings": null}',
    '{"verdict": "YES", "findings": []}',
])
def test_a_clean_review_is_not_a_contract_violation(payload):
    """The regression the shape guard could most easily cause.

    A persona that found nothing omits ``findings`` and votes YES. Reading that
    as malformed would cost a second paid call, discard the vote, and make
    CONVERGED unreachable for a healthy panel — see
    ``tests/engine/test_seam_isolation.py``, which pins the loop half.
    """
    report = parse_findings("p", _reply(payload))

    assert report.findings == []
    assert report.verdict == "YES"
    assert not _is_parse_failure(report)


def test_a_malformed_entry_loses_itself_not_the_reply():
    report = parse_findings("p", _reply(json.dumps({"verdict": "NO", "findings": [
        {"title": "a real finding", "severity": "BLOCKER"},
        "just a sentence",
        None,
        {"title": "another real one"}]})))

    titles = [f.title for f in report.findings]
    assert "a real finding" in titles and "another real one" in titles
    dropped = [f for f in report.findings if f.claim_class == "meta"]
    assert len(dropped) == 1
    assert "2" in dropped[0].title, dropped[0].title
    assert report.verdict == "NO"        # the rest of the reply still counts


def test_a_dropped_entry_cannot_reset_the_stall_counter():
    """The record of a dropped claim must not read as new material."""
    report = parse_findings("p", _reply('{"findings": ["nonsense"]}'))

    assert report.findings[0].severity is Severity.NOTE


@pytest.mark.parametrize("raw", [["UGLY"], ["UGLY", "BLOCKER"], {"level": "UGLY"}])
def test_a_non_scalar_severity_is_unresolved_not_stringified(raw):
    """``str(["UGLY"])`` used to resolve to UGLY by accident of punctuation."""
    report = parse_findings("p", _one_finding(severity=raw))

    assert report.findings[0].severity is UNRESOLVED_SEVERITY
    assert report.unresolved_severities == [repr(raw)[:120]]


def test_a_non_string_file_is_coerced():
    report = parse_findings("p", _one_finding(file={"path": "a.py"}, line=3))

    assert report.findings[0].file == "{'path': 'a.py'}"


# --- one extractor for one contract (#19) -----------------------------------

def test_the_clusterer_shares_the_extractor_and_the_notion_of_verbatim():
    """Two parsers for one contract meant a fix to one silently left the other.

    Extended for #61: the clusterer grew an echo detector of its own, so it now
    shares the whitespace normalisation as well. Two notions of "verbatim" would
    be the same defect this test was written for, one function along.
    """
    from kuang.backends.claude_code import cluster, contract

    assert cluster.extract_json_blocks is contract.extract_json_blocks
    assert cluster._collapse_whitespace is contract._collapse_whitespace


# --- an echoed output contract (#26) -----------------------------------------

def test_an_echoed_template_does_not_discard_the_real_review():
    """REGRESSION for #26: the template parsed, so ``blocks[-1]`` took it.

    Both halves of the defect are here: the persona's findings came back empty
    and the literal placeholder ``'YES | NO'`` was counted as a YES.
    """
    real = _reply('{"verdict": "NO", "findings": [{"title": "unbounded retry", '
                  '"severity": "UGLY", "claim_class": "ruin", "file": "r.py", '
                  '"line": 12, "evidence": "read it"}]}')

    report = parse_findings("adversary", real + "\nFor reference, the contract was:\n"
                            + _reply(_TEMPLATE_BLOCK))

    assert [f.title for f in report.findings if f.claim_class != "meta"] == [
        "unbounded retry"]
    assert report.verdict == "NO"
    assert report.unresolved_severities == []


def test_the_whole_contract_pasted_back_does_not_discard_the_review():
    """The other echo shape: the contract restated in full, prose included.

    It used to behave differently only by accident — the contract's own prose
    named a fenced ``json`` block *inline*, so the extraction regex opened on
    that mention and closed on the template's fence, capturing a fragment of
    prose. We wrote that trip hazard ourselves; the contract no longer contains
    it, so this collapses into the case above.
    """
    real = _reply('{"verdict": "NO", "findings": [{"title": "real", '
                  '"severity": "BLOCKER", "claim_class": "logic"}]}')

    report = parse_findings("adversary", real + "\n" + FINDINGS_CONTRACT)

    assert [f.title for f in report.findings if f.claim_class != "meta"] == ["real"]
    assert report.verdict == "NO"


def test_a_reply_that_is_only_an_echo_is_a_contract_violation():
    """No review to recover, so it takes the retry path — and casts no vote."""
    report = parse_findings("p", "Understood. " + _reply(_TEMPLATE_BLOCK))

    assert _is_parse_failure(report)
    assert report.verdict is None
    assert "echo" in report.findings[0].title


def test_the_shipped_contract_echoed_whole_is_recognised():
    """Anti-drift, end to end: the detector is fed the contract we actually ship.

    It goes red if the template in the prompt and the block the detector knows
    ever part company — which is why they are one constant, not two.
    """
    report = parse_findings("p", "Here is the contract:\n" + FINDINGS_CONTRACT)

    assert _is_parse_failure(report)
    assert report.verdict is None


def test_the_shipped_contract_carries_exactly_the_block_the_detector_knows():
    assert FINDINGS_CONTRACT.count("```json") == 1
    assert _TEMPLATE_BLOCK in FINDINGS_CONTRACT


def test_an_ignored_echo_is_recorded_but_cannot_read_as_new_material():
    """Said out loud (#24's doctrine), at NOTE so it cannot reset the stall counter."""
    real = _reply('{"verdict": "NO", "findings": [{"title": "real", '
                  '"severity": "BLOCKER", "claim_class": "logic"}]}')

    report = parse_findings("p", real + _reply(_TEMPLATE_BLOCK))

    echoes = [f for f in report.findings if f.claim_class == "meta"]
    assert len(echoes) == 1
    assert "echo" in echoes[0].title
    assert echoes[0].severity is Severity.NOTE


def test_a_block_that_only_resembles_the_template_still_wins():
    """The skip rule is exact on purpose: only text we shipped is ever skipped.

    A paraphrase is not silently guessed at — it wins its block, and the verdict
    guard is what stops it voting. Widening this into "the last block that looks
    plausible" is how an earlier, stale block gets promoted over a real one.
    """
    real = _reply('{"verdict": "NO", "findings": [{"title": "real", '
                  '"severity": "BLOCKER", "claim_class": "logic"}]}')
    near = _reply('{"verdict": "YES | NO", "findings": [{"title": "...", '
                  '"severity": "NOTE | NON_BLOCKING | BLOCKER | UGLY"}]}')

    report = parse_findings("p", real + near)

    assert "real" not in [f.title for f in report.findings]
    assert report.verdict is None                    # ...but it does not vote
    assert report.unresolved_verdict == "YES | NO"


# --- the verdict is categorical (#26) ----------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("YES", "YES"),
    ("yes", "YES"),
    ("  YES  ", "YES"),
    ("YES.", "YES"),                              # trailing punctuation, one word
    ("'NO'", "NO"),
    ("no", "NO"),
])
def test_one_word_from_the_vocabulary_is_the_vote(raw, expected):
    report = parse_findings("p", _reply(json.dumps({"verdict": raw, "findings": []})))

    assert report.verdict == expected
    assert report.unresolved_verdict is None


@pytest.mark.parametrize("raw", [
    "YES | NO",                                   # the contract template, echoed
    "YES/NO",
    "YES, no issues found",                       # prose, however affirmative
    "YES — nothing blocking",
    "SOUND-WITH-CONCERNS",
    "MAYBE", "NO UNLESS", "APPROVE", "LGTM", "Y",
    "", "   ",
])
def test_anything_that_is_not_one_of_the_two_words_does_not_vote(raw):
    """No prose is read. The vote surface is two literals and nothing else.

    Both halves matter: it must not silently count, and it must not silently
    *not*-count either — the raw value is kept verbatim for a human.
    """
    report = parse_findings("p", _reply(json.dumps({"verdict": raw, "findings": []})))

    assert report.verdict is None
    assert report.unresolved_verdict == raw


def test_an_absent_verdict_records_nothing():
    """The boundary: a persona that claimed no verdict is not one we misread."""
    report = parse_findings("p", _reply('{"findings": []}'))

    assert report.verdict is None
    assert report.unresolved_verdict is None


def test_a_null_verdict_records_nothing():
    report = parse_findings("p", _reply('{"verdict": null, "findings": []}'))

    assert report.verdict is None
    assert report.unresolved_verdict is None


@pytest.mark.parametrize("raw, expected", [
    ('{"x": 1}', "{'x': 1}"),
    ('3', "3"),
    ('["YES"]', "['YES']"),
    ('true', "True"),
])
def test_a_non_string_verdict_is_unresolved_and_recorded(raw, expected):
    """It used to reach the loop's quorum count as a string and vote on ``str()``.

    ``['YES']`` resolved by accident of punctuation, exactly as ``["UGLY"]`` did
    for severity (#25). The loop keeps its own ``str()`` guard, tested against a
    backend that returns a non-string verdict directly — see
    ``tests/engine/test_quorum.py``.
    """
    report = parse_findings("p", _reply('{"verdict": %s, "findings": []}' % raw))

    assert report.verdict is None
    assert report.unresolved_verdict == expected


def test_a_recorded_raw_verdict_is_capped():
    """A model-controlled string reaches the run artefact; bound it there."""
    report = parse_findings("p", _reply(json.dumps({"verdict": "Z" * 500})))

    assert report.unresolved_verdict == "Z" * 120


def test_the_contract_asks_for_the_verdict_the_parser_reads():
    """The prompt side, because a categorical field we never explained is a trap."""
    assert "exactly one word" in FINDINGS_CONTRACT
    assert "no BLOCKER and no UGLY" in FINDINGS_CONTRACT


def test_the_extractor_returns_every_fenced_block():
    from kuang.backends.claude_code.contract import extract_json_blocks

    assert extract_json_blocks(_reply("A") + _reply("B")) == ["A\n", "B\n"]
    assert extract_json_blocks("no block here") == []


def test_an_echo_before_the_real_block_is_not_reported():
    """Only echoes actually skipped are counted; the last block still wins."""
    real = _reply('{"verdict": "NO", "findings": [{"title": "real", '
                  '"severity": "BLOCKER", "claim_class": "logic"}]}')

    report = parse_findings("p", _reply(_TEMPLATE_BLOCK) + real)

    assert [f.title for f in report.findings] == ["real"]
    assert report.verdict == "NO"


def test_an_echo_after_an_empty_review_does_not_take_the_retry():
    """Converted (#30). This test pinned the retry firing here; it no longer does.

    The old ``_is_parse_failure`` INFERRED "unreadable" from the shape of a report
    — no verdict, one finding, ``claim_class == "meta"`` — and an empty payload
    followed by a contract echo happens to wear that shape. It was pinned as
    benign. It is not: this reply *was* readable, and #68 measured what the
    reformatter does when fed text that is not a review — four times out of four
    it returned a fabricated ``BLOCKER`` attributed to a reviewer that never
    raised one. Paying a second call to reformat an empty review is buying that
    risk for nothing.

    ``_is_parse_failure`` now reads the status ``parse_findings`` SET, so a
    readable empty review is ``FOUND_NOTHING`` and stops here.
    """
    report = parse_findings("p", _reply('{"findings": []}') + _reply(_TEMPLATE_BLOCK))

    assert not _is_parse_failure(report)
    assert report.status is PersonaStatus.FOUND_NOTHING
    # The echo is still recorded, exactly as before — only the retry is gone.
    assert [f.claim_class for f in report.findings] == ["meta"]


def test_a_review_whose_only_finding_is_labelled_meta_is_still_a_review():
    """The false retry the old inferred predicate bought, stated directly.

    A persona may write ``claim_class: "meta"`` and claim no verdict. That is a
    review, and it used to be indistinguishable from a reply we could not read.
    """
    report = parse_findings("p", _reply(
        '{"findings": [{"title": "an observation", "severity": "NOTE", '
        '"claim_class": "meta"}]}'))

    assert not _is_parse_failure(report)
    assert report.status is PersonaStatus.CONTRIBUTED


# --- whose finding is it: the parser's, or the persona's? (#73) --------------
#
# Every diagnosis this module produces is a finding about the RUN, not about the
# change under review, and it says so at the point it is built. The flag exists
# because the obvious alternative — matching ``claim_class == "meta"`` in the
# consumer — reads a model-controlled string: measured, excluding that string from
# the ledger turns a run that should halt ``escalate_ugly`` on a persona-declared
# ``meta`` UGLY into one that halts ``converged``. The last two rows below are the
# ones that keep this test honest; without them, "everything is about the run"
# would satisfy it.

@pytest.mark.parametrize("reply, title_starts, about_run", [
    # The five diagnoses this module produces, one row each.
    ("I could not comply.", "no structured output", True),
    (_reply("{not json at all"), "unparseable JSON findings", True),
    (_reply('"all good"'), "findings contract violated", True),
    (_reply('{"findings": "none"}'), "findings contract violated", True),
    (_reply('{"findings": ["a string", {"title": "real"}]}'),
     "1 malformed finding entry discarded", True),
    (_reply(_TEMPLATE_BLOCK), "findings contract violated", True),
    (_reply('{"verdict": "NO", "findings": [{"title": "real defect", '
            '"severity": "BLOCKER", "claim_class": "logic"}]}')
     + _reply(_TEMPLATE_BLOCK), "output contract echoed", True),
    # ...and the persona's own claims, which are never about the run whatever
    # they are labelled. A finding about a metaclass is naturally called `meta`.
    (_reply('{"verdict": "NO", "findings": [{"title": "real defect", '
            '"severity": "BLOCKER", "claim_class": "logic"}]}'),
     "real defect", False),
    (_reply('{"verdict": "NO", "findings": [{"title": "metaclass mutates state", '
            '"severity": "UGLY", "claim_class": "meta"}]}'),
     "metaclass mutates state", False),
], ids=["no-json-block", "unparseable-json", "payload-not-an-object",
        "findings-not-a-list", "malformed-entry", "echo-only",
        "real-plus-echo", "persona-real-finding", "persona-labelled-meta"])
def test_who_a_finding_is_about_is_set_where_it_is_built(reply, title_starts,
                                                         about_run):
    report = parse_findings("p", reply)
    match = [f for f in report.findings if f.title.startswith(title_starts)]

    assert match, [f.title for f in report.findings]
    assert match[0].about_run is about_run
