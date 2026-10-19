# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A persona file that cannot be read or decoded is reported, never a traceback (#104).

Before #104, three of the four sourcing paths read with no guard that covered a decode
failure: ``CLAUDE.md`` with no guard at all, and ``panel.md`` and a markdown ``--panel``
behind an ``except OSError`` that ``UnicodeDecodeError`` (a ``ValueError``) escapes. Each
took the run down with a traceback naming ``personas.py`` rather than the file. The same
unguarded ``CLAUDE.md`` read crashed on a directory or an unreadable file, stopped a good
``panel.md`` from ever being consulted, and hung on a FIFO. And every read decoded with
the LOCALE's encoding, so a valid UTF-8 file sourced mojibake silently under a Latin-1
locale (C3, ruled a member of #40's first-criterion class).

The rulings this pins (all on #104):

* an undecodable or unreadable ``CLAUDE.md`` is ``Discarded("CLAUDE.md", "malformed",
  …)`` and sourcing carries on to ``panel.md`` — #16's split, unchanged;
* the three persona reads decode as UTF-8 explicitly, so a Latin-1 file is a
  ``malformed`` discard naming the encoding (C8, accepted);
* the implicit tiers require a regular file; ``--panel`` keeps #59's rule, so process
  substitution still works and a FIFO with no writer still blocks there.

What a detail may carry is #105's rule, unchanged: a type and a coordinate, never
content. For a decode failure the coordinate is a byte offset and the codec is ours.

**Which are regressions, measured on main with this file in place.** Recorded in the PR
body; any test green on main says so in its own docstring.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from pathlib import Path

import pytest

from kuang.backends.claude_code import personas as personas_module
from kuang.backends.claude_code.personas import PanelError, load_personas

# Valid UTF-8 the parser commits to, carrying everything a locale decode garbles: an
# emoji before a name, an em-dash separator, a hyphen separator, and an accented
# letter in a grounding.
_UTF8_EXPERTS = """\
# A repo

# Resident Experts

## 🎯 The Analyst — Correctness

Does it compute the right thing? Café arithmetic.

## 🗡️ The Adversary - Threats

Break it.
"""

_GOOD_PANEL_MD = """\
# Resident Experts

## Auditor — Constraints

What external rules must this not violate?
"""

# Not UTF-8 at byte 3. The ASCII before it is a stand-in for a repository's content,
# which a detail must never carry (#105).
_UNDECODABLE = b"ACM\xff\xfe secret-internal-name\n"
_UNDECODABLE_DETAIL = "UnicodeDecodeError at byte 3: not utf-8 text"

# A1's shape with a readable heading: the declaration is visible to a human and the
# file still cannot be decoded.
_UNDECODABLE_WITH_HEADING = (b"# Resident Experts\n\n## The Auditor \xe2\x80\x94 R\xe9gles"
                             b"\n\nRules.\n")

# C8: a Latin-1 file that sources under a Latin-1 locale on main.
_LATIN1_EXPERTS = "# Resident Experts\n\n## The Auditor - Règles\n\nRules.\n".encode(
    "latin-1")

_NOT_REGULAR = "not a regular file"


def _write(repo: Path, name: str, content: str | bytes) -> Path:
    path = repo / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _stub_yaml(monkeypatch, payload=None):
    """A ``yaml`` whose ``safe_load`` returns ``payload``, so the YAML branch is reached
    hermetically wherever the optional extra is absent."""
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda _text: payload
    monkeypatch.setitem(sys.modules, "yaml", stub)


def _discards(source) -> list[tuple[str, str, str]]:
    return [(d.origin, d.reason, d.detail) for d in source.discarded]


def _source_in_thread(repo: Path, fifos: list[Path]):
    """``load_personas`` on a thread, so a blocking open fails the test, not the suite.

    A FIFO with no writer blocks ``open()`` for reading. If sourcing opens one, the
    join times out, and a non-blocking writer is opened on each FIFO to release the
    reader with EOF before the test fails. The thread is a daemon besides.
    """
    result: dict = {}

    def _run():
        try:
            result["value"] = load_personas(repo)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the test thread
            result["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=10)
    blocked = worker.is_alive()
    if blocked:
        for fifo in fifos:
            try:
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass
        worker.join(timeout=10)
    assert not blocked, "sourcing opened a FIFO named by an implicit tier and blocked"
    if "error" in result:
        raise result["error"]
    return result["value"]


# --- every implicit tier: an undecodable file is a discard, and the run goes on -------

@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md", "panel.yaml"])
def test_an_undecodable_implicit_declaration_is_a_malformed_discard(
        tmp_path, monkeypatch, tier):
    """REGRESSION for #104 (A1, A5). ``panel.yaml`` is red on main on the DETAIL only.

    The YAML branch's broad ``except`` already caught the decode failure on main (A7),
    recording the type alone. It is parametrised beside the other two because "the
    same on all four sourcing paths" is the acceptance, and the detail it now carries is
    the shared rule's, not the YAML branch's own.
    """
    _stub_yaml(monkeypatch)
    _write(tmp_path, tier, _UNDECODABLE)

    roster, source, _ = load_personas(tmp_path)
    names = [p.name for p in roster]

    assert source.label == "default"
    assert _discards(source) == [(tier, "malformed", _UNDECODABLE_DETAIL)]
    assert "correctness" in names, "the run carries on with the default panel"


def test_an_undecodable_panel_yaml_without_the_extra_is_still_unsupported(
        tmp_path, monkeypatch):
    """PAIR, green on main (A6): the extra is checked before the file is read.

    A ``panel.yaml`` that could never have been parsed says *install the extra*, which
    is #16's split; reading it first would turn that into *fix the file*.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    _write(tmp_path, "panel.yaml", _UNDECODABLE)

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [("panel.yaml", "unsupported", "ModuleNotFoundError")]


def test_an_undecodable_claude_md_with_a_readable_heading_is_a_discard(tmp_path):
    """REGRESSION for #104 (A2): the heading a human can see is not read as clean."""
    _write(tmp_path, "CLAUDE.md", _UNDECODABLE_WITH_HEADING)

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [
        ("CLAUDE.md", "malformed", "UnicodeDecodeError at byte 40: not utf-8 text")]


def test_a_utf16_claude_md_is_a_discard_naming_utf8(tmp_path):
    """REGRESSION for #104 (A3): a file saved as UTF-16 is not UTF-8, and says so.

    The ruling's accepted cost: a UTF-16 ``CLAUDE.md`` warns even if it declared no
    panel, because a file that cannot be decoded cannot show that it declared nothing.
    """
    _write(tmp_path, "CLAUDE.md", _UTF8_EXPERTS.encode("utf-16"))

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [
        ("CLAUDE.md", "malformed", "UnicodeDecodeError at byte 0: not utf-8 text")]


def test_a_latin1_claude_md_is_a_discard_whatever_the_locale(tmp_path):
    """REGRESSION for #104 (A4) — and C8, the accepted mirror image.

    Under a Latin-1 locale this file sourced on main, because the locale decoded it.
    Persona files are read as UTF-8 by ruling, so it becomes a discard that names the
    encoding, and the operator can re-save it. In-process this asserts the decode is
    explicit: under the suite's UTF-8 locale, main raised instead.
    """
    _write(tmp_path, "CLAUDE.md", _LATIN1_EXPERTS)

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [
        ("CLAUDE.md", "malformed", "UnicodeDecodeError at byte 38: not utf-8 text")]


# --- a byte-order mark (found while implementing; folded in by ruling) ----------------

_BOM = "﻿"
_HEADING_FIRST = "# Resident Experts\n\n## Auditor — Constraints\n\nRules for Café.\n"


@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md"])
def test_a_byte_order_mark_does_not_hide_a_line_one_heading(tmp_path, tier):
    """REGRESSION for #104 (the BOM rows, ruled in): a declared panel was lost.

    The heading regex anchors at the start of a line, and a decoded UTF-8 byte-order
    mark is U+FEFF, so line 1 began with the mark and not ``#``. On main a
    ``CLAUDE.md`` saved with one sourced the default panel with nothing discarded and
    no warning; a ``panel.md`` was reported as naming nobody.
    """
    _write(tmp_path, tier, _BOM + _HEADING_FIRST)

    roster, source, _ = load_personas(tmp_path)

    assert roster[0].name == "Auditor"
    assert source.discarded == ()


def test_a_byte_order_mark_does_not_hide_an_explicit_panel(tmp_path):
    """REGRESSION for #104 (the BOM rows): ``--panel`` refused it as naming nobody."""
    path = _write(tmp_path, "x.md", _BOM + _HEADING_FIRST)

    roster, source, _ = load_personas(tmp_path, panel_path=path)

    assert source.label == "--panel"
    assert roster[0].name == "Auditor"


def test_a_file_with_a_byte_order_mark_sources_what_one_without_does(tmp_path):
    """PAIR, red on main: the mark is removed and nothing else is. The roster, names
    and groundings byte for byte, are the BOM-less file's — so a strip that ate the
    first character, or the whole first line, fails here, and so does one that removed
    a U+FEFF from the body, where it is a character of the file and not a mark."""
    with_bom, without = tmp_path / "with", tmp_path / "without"
    with_bom.mkdir()
    without.mkdir()
    body = _HEADING_FIRST + "Zero\ufeffwidth.\n"
    _write(with_bom, "CLAUDE.md", _BOM + body)
    _write(without, "CLAUDE.md", body)

    def _sourced(repo):
        roster, source, _ = load_personas(repo)
        return source, [(p.name, p.grounding) for p in roster]

    assert _sourced(with_bom) == _sourced(without)
    # Equality alone cannot see a strip that removed the body's U+FEFF from BOTH
    # files (found by mutation: that row survived), so the body is asserted too.
    assert "Zero﻿width." in _sourced(with_bom)[1][0][1]


def test_every_leading_byte_order_mark_is_removed_not_only_the_first(tmp_path):
    """REGRESSION for #104 (the doubled-BOM ruling). A file that two tools each saved
    with a mark carries two, and removing only the first left the second hiding the
    line-1 heading, so the declared panel was lost exactly as with one. Kills a
    one-mark ``removeprefix``; the pair above forbids a strip that reaches the body."""
    _write(tmp_path, "CLAUDE.md", _BOM + _BOM + _HEADING_FIRST)

    roster, source, _ = load_personas(tmp_path)

    assert roster[0].name == "Auditor"
    assert source.discarded == ()


def test_a_decode_offset_counts_the_byte_order_mark(tmp_path):
    """REGRESSION for #104 on main; guards the REJECTED ``utf-8-sig`` decode.

    ``utf-8-sig`` reports a decode error's offset after the mark it consumed, so the
    operator would be sent three bytes short. The mark is stripped after a plain UTF-8
    decode, and the offset is the file's own.
    """
    _write(tmp_path, "panel.md", b"\xef\xbb\xbf" + _UNDECODABLE)

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [
        ("panel.md", "malformed", "UnicodeDecodeError at byte 6: not utf-8 text")]


# --- A11, and its pair: precedence is otherwise untouched ----------------------------

def test_a_broken_claude_md_does_not_stop_a_good_panel_md(tmp_path):
    """REGRESSION for #104 (A11): the good panel used to be unreachable.

    ``CLAUDE.md`` is consulted first. On main its read raised before ``panel.md`` was
    looked at; now its discard is recorded beside the panel that was used.
    """
    _write(tmp_path, "CLAUDE.md", _UNDECODABLE)
    _write(tmp_path, "panel.md", _GOOD_PANEL_MD)

    roster, source, _ = load_personas(tmp_path)

    assert source.label == "panel file"
    assert [p.name for p in roster][0] == "Auditor"
    assert _discards(source) == [("CLAUDE.md", "malformed", _UNDECODABLE_DETAIL)]


def test_a_good_claude_md_still_wins_over_panel_md(tmp_path):
    """PAIR, green on main: guards precedence against the A11 fix reordering it."""
    _write(tmp_path, "CLAUDE.md", _UTF8_EXPERTS)
    _write(tmp_path, "panel.md", _GOOD_PANEL_MD)

    roster, source, _ = load_personas(tmp_path)

    assert source.label == "CLAUDE.md"
    assert [p.name for p in roster][:2] == ["The Analyst", "The Adversary"]
    assert source.discarded == ()


# --- clean versus not clean ----------------------------------------------------------

def test_a_valid_utf8_claude_md_with_no_heading_still_reads_clean(tmp_path):
    """PAIR, green on main: non-ASCII content and no declaration is a healthy run.

    The mirror image of every discard above: reading as UTF-8 must not make a file
    with em-dashes, emoji and accents, and no experts heading, report anything.
    """
    _write(tmp_path, "CLAUDE.md", "# Café — notes 🎯\n\nNo panel here.\n")

    source = load_personas(tmp_path)[1]

    assert source.label == "default"
    assert source.discarded == ()


def test_a_valid_utf8_declaration_sources_every_character_intact(tmp_path):
    """PAIR, green on main under a UTF-8 locale: the explicit decode changes nothing
    for a healthy file. The emoji is stripped from the name, as it always was, and the
    grounding keeps its accent."""
    _write(tmp_path, "CLAUDE.md", _UTF8_EXPERTS)

    roster, source, _ = load_personas(tmp_path)

    assert source.label == "CLAUDE.md"
    assert [p.name for p in roster][:2] == ["The Analyst", "The Adversary"]
    assert "Café arithmetic." in roster[0].grounding
    assert roster[0].grounding.startswith("You are The Analyst — Correctness.")


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                    reason="needs a POSIX permission bit that binds this user")
@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md"])
def test_an_unreadable_implicit_declaration_is_a_discard(tmp_path, tier):
    """REGRESSION for #104 (B2) — ``panel.md`` green on main (B5)."""
    path = _write(tmp_path, tier, _GOOD_PANEL_MD)
    path.chmod(0)
    try:
        source = load_personas(tmp_path)[1]
    finally:
        path.chmod(0o600)

    assert _discards(source) == [(tier, "malformed", "PermissionError")]


@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md", "panel.yaml"])
def test_a_directory_on_an_implicit_tier_is_not_a_regular_file(
        tmp_path, monkeypatch, tier):
    """REGRESSION for #104 (B1). ``panel.md`` and ``panel.yaml`` are red on main on the
    DETAIL only: they were already discarded (B4), as ``IsADirectoryError``."""
    _stub_yaml(monkeypatch)
    (tmp_path / tier).mkdir()

    source = load_personas(tmp_path)[1]

    assert _discards(source) == [(tier, "malformed", _NOT_REGULAR)]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md", "panel.yaml"])
def test_a_fifo_on_an_implicit_tier_is_discarded_not_waited_on(
        tmp_path, monkeypatch, tier):
    """REGRESSION for #104 (B3, B6): a FIFO with no writer hung the run.

    A FIFO named ``CLAUDE.md`` or ``panel.md`` in a repository is never process
    substitution, so the implicit tiers require a regular file and fire on no healthy
    run. ``--panel`` keeps #59's rule; see the process-substitution pair below.
    """
    _stub_yaml(monkeypatch)
    fifo = tmp_path / tier
    os.mkfifo(fifo)

    source = _source_in_thread(tmp_path, [fifo])[1]

    assert _discards(source) == [(tier, "malformed", _NOT_REGULAR)]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_symlink_to_a_regular_file_is_still_read(tmp_path):
    """PAIR, green on main: the regular-file check follows a link, as ``open`` does.

    A ``CLAUDE.md`` that is a symlink to a real file is a healthy repository layout, and
    must not be reported as "not a regular file".
    """
    real = _write(tmp_path, "real-claude.md", _UTF8_EXPERTS)
    (tmp_path / "CLAUDE.md").symlink_to(real)

    source = load_personas(tmp_path)[1]

    assert source.label == "CLAUDE.md"
    assert source.discarded == ()


# --- --panel: refuses, through the message it already had -----------------------------

@pytest.mark.parametrize("name", ["x.md", "x.yaml"])
def test_an_undecodable_explicit_panel_refuses_naming_the_file(
        tmp_path, monkeypatch, name):
    """REGRESSION for #104 (A9) — ``x.yaml`` refused on main with the type alone (A10)."""
    _stub_yaml(monkeypatch)
    path = _write(tmp_path, name, _UNDECODABLE)

    with pytest.raises(PanelError) as caught:
        load_personas(tmp_path, panel_path=path)

    assert str(caught.value) == (f"--panel {path}: it could not be read or parsed "
                                 f"({_UNDECODABLE_DETAIL})")


def test_an_explicit_panel_naming_a_directory_is_refused_as_before(tmp_path):
    """PAIR, green on main (B7): ``--panel`` makes no regular-file check, so its detail
    is still the read's own exception type."""
    (tmp_path / "pd").mkdir()

    with pytest.raises(PanelError) as caught:
        load_personas(tmp_path, panel_path=tmp_path / "pd")

    assert str(caught.value) == (f"--panel {tmp_path / 'pd'}: it could not be read or "
                                 f"parsed (IsADirectoryError)")


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd")
def test_process_substitution_still_sources_an_explicit_panel(tmp_path):
    """PAIR, green on main: ``--panel <(…)`` is a pipe named ``/dev/fd/N``.

    It is not a regular file, so a regular-file check on ``--panel`` would break it.
    #59's rule: an explicit non-regular file is read, and one with no writer blocks
    before any spend — a stated under-report, not refused.
    """
    r, w = os.pipe()
    try:
        os.write(w, _GOOD_PANEL_MD.encode("utf-8"))
        os.close(w)
        roster, source, _ = load_personas(tmp_path, panel_path=f"/dev/fd/{r}")
    finally:
        os.close(r)

    assert source.label == "--panel"
    assert roster[0].name == "Auditor"


# --- closed by default ---------------------------------------------------------------

@pytest.mark.parametrize("raised", [ValueError("a bug, not a bad file"),
                                    RuntimeError("a bug, not a bad file")],
                         ids=["ValueError", "RuntimeError"])
@pytest.mark.parametrize("tier", ["CLAUDE.md", "panel.md"])
def test_a_fault_the_read_does_not_name_stays_loud(tmp_path, monkeypatch, tier, raised):
    """PAIR, green on main: guards the REJECTED catches, ``(OSError, ValueError)`` and
    ``Exception``.

    ``UnicodeDecodeError`` is a ``ValueError``, so catching ``ValueError`` around the
    read would cover every discard above — and would also record a bug as the
    operator's malformed file, with the traceback that would have found it gone. Only
    the two faults a read can name, ``OSError`` and ``UnicodeDecodeError``, are caught.
    """
    _write(tmp_path, tier, _GOOD_PANEL_MD)

    def _broken(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise raised

    monkeypatch.setattr(Path, "read_text", _broken)

    with pytest.raises(type(raised), match="a bug, not a bad file"):
        load_personas(tmp_path)


def test_a_bug_in_the_markdown_parser_stays_loud(tmp_path, monkeypatch):
    """PAIR, green on main: the catch covers the READ, not the parse after it.

    The markdown branch's parse is ours; an exception from it is a bug, never a
    ``malformed`` file. (The YAML branch's validation is ours too and is closed the
    same way since #115; only pyyaml's own parse is caught broadly.)
    """
    _write(tmp_path, "panel.md", _GOOD_PANEL_MD)

    def _broken(_text):
        raise ValueError("a bug in the parser")

    monkeypatch.setattr(personas_module, "parse_claude_md_experts", _broken)

    with pytest.raises(ValueError, match="a bug in the parser"):
        load_personas(tmp_path)


def test_a_decode_detail_never_carries_the_file(tmp_path):
    """REGRESSION for #104 (A5), held to #105's rule: on main the read tracebacked.

    ``str(UnicodeDecodeError)`` names the codec's reason and the offending byte. The
    record carries a type, a byte offset and our own codec name, and nothing the file
    said. Exact equality, so any rendering of the exception — or its ``reason`` — fails.
    """
    _write(tmp_path, "panel.md", _UNDECODABLE)

    detail = load_personas(tmp_path)[1].discarded[0].detail

    assert detail == _UNDECODABLE_DETAIL
    assert "secret" not in detail and "0xff" not in detail and "invalid" not in detail
