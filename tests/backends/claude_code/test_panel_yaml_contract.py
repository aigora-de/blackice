# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""A ``panel.yaml`` entry of the wrong type is refused or discarded, never run (#115).

``_declared_persona`` built a ``Persona`` from whatever ``pyyaml`` handed it, and a
dataclass checks nothing. So an ill-typed entry did one of three things, all measured
on main through the real CLI before this file was written:

* **tracebacked** at rc 1 — a ``name`` or ``grounding`` that is not a string died in
  ``_ensure_specialists``, outside every ``try``; ``tools:`` null and ``tools: [1]``
  died in ``permissions`` after the first ``[panel]`` line had printed;
* **ran silently with a grant it was never given** — ``tools: Read`` was split into
  ``--allowedTools R e a d``, a mapping into its keys, and two entries sharing a name
  ran the first persona with the second's tools while the console and the artefact said
  the first's;
* **blamed the spawn for the operator's file** — ``model: 42`` reached argv, and that
  persona was ``spawn_failed`` in every epoch while a live run paid for the rest.

The contract, as ruled on #115: ``name`` a printable string, non-empty after
stripping; ``grounding`` a string, or absent or null (#103's "no mandate"); ``tools`` a
list of strings, absent for the default or ``[]`` for none; ``model`` a non-empty
string, or absent or null for the default; names unique within the file. One ill-typed
entry discards the whole file as ``malformed`` (implicit) or refuses it (``--panel``),
and the detail is a coordinate plus our own type names, never the value (#105).

Every test here uses the REAL parser, because the shapes are pyyaml's (a bare date is a
``datetime.date``, ``grounding:`` is ``None``) and a stub asserting a shape nobody
measured tests the stub. The module skips where the optional extra is absent, which is
this repo's venv, and runs in CI's ``[yaml]`` job.

**Which are regressions, measured on main with this file in place.** Recorded in the PR
body; any test green on main says so in its own docstring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from kuang.backends.claude_code import personas as personas_module  # noqa: E402
from kuang.backends.claude_code.permissions import DEFAULT_ALLOWED_TOOLS  # noqa: E402
from kuang.backends.claude_code.personas import (PanelError,  # noqa: E402
                                                 load_personas,
                                                 parse_claude_md_experts)

_GOOD = "  - name: Analyst\n    grounding: Does it compute the right thing?\n"


def _entry(field: str) -> str:
    """A file whose second entry carries ``field``, beside a well-typed first one.

    The bad entry is second so every detail's coordinate is ``personas[1]`` — an index
    that a validator counting from the wrong place, or reporting the first entry it
    saw, would get wrong.
    """
    return f"personas:\n{_GOOD}  - name: Sentinel\n    grounding: Hunt ruin.\n{field}"


# Each ill-typed shape measured on main, and the detail it is now recorded with. The
# row ids are the correction's on #115.
_ILL_TYPED = [
    ("T1", "personas:\n  - name:\n", "personas[0].name: null, not a string"),
    ("T2", "personas:\n  - name: 42\n", "personas[0].name: int, not a string"),
    ("T3", "personas:\n  - name: [a, b]\n", "personas[0].name: list, not a string"),
    ("T4", "personas:\n  - name: true\n", "personas[0].name: bool, not a string"),
    ("T7", "personas:\n  - name: {a: b}\n", "personas[0].name: dict, not a string"),
    ("T5", 'personas:\n  - name: ""\n', "personas[0].name: empty"),
    ("T6", 'personas:\n  - name: "   "\n', "personas[0].name: empty"),
    ("N2", 'personas:\n  - name: "A\\nB"\n',
     "personas[0].name: a non-printable character"),
    ("G1", _entry("    grounding: 42\n"), None),
    ("G2", _entry("    grounding: [a]\n"), None),
    ("G3", _entry("    grounding: true\n"), None),
    ("G4", _entry("    grounding: {a: b}\n"), None),
    ("G5", _entry("    grounding: 2024-01-01\n"), None),
    ("W1", _entry("    tools: Read\n"), "personas[1].tools: str, not a list of strings"),
    ("W2", _entry("    tools:\n"), "personas[1].tools: null, not a list of strings"),
    ("W3", _entry("    tools: [Read, 1]\n"), "personas[1].tools[1]: int, not a string"),
    ("W6", _entry("    tools: {a: b}\n"), "personas[1].tools: dict, not a list of strings"),
    ("M1", _entry("    model: 42\n"), "personas[1].model: int, not a string"),
    ("M2", _entry("    model: [a]\n"), "personas[1].model: list, not a string"),
    ("M5", _entry("    model: true\n"), "personas[1].model: bool, not a string"),
    ("M3", _entry('    model: ""\n'), "personas[1].model: empty"),
    # A character argv cannot carry, from a YAML escape: each sourced on main and then
    # failed that persona's spawn at rc 0 (X1-X6), or split the dry run's argv line (X7,
    # X8).
    ("X1", _entry('    model: "m\\0x"\n'),
     "personas[1].model: a non-printable character"),
    ("X2", _entry('    model: "m\\ud800"\n'),
     "personas[1].model: a non-printable character"),
    ("X8", _entry('    model: "m\\nx"\n'), "personas[1].model: a non-printable character"),
    ("X3", _entry('    tools: [Read, "Re\\0ad"]\n'),
     "personas[1].tools[1]: a non-printable character"),
    ("X4", _entry('    tools: ["\\udfff"]\n'),
     "personas[1].tools[0]: a non-printable character"),
    ("X7", _entry('    tools: ["Read\\nBash"]\n'),
     "personas[1].tools[0]: a non-printable character"),
    ("X5", _entry('  - name: Critic\n    grounding: "real\\0mandate"\n'),
     "personas[2].grounding: a character a command line cannot carry"),
    ("X6", _entry('  - name: Critic\n    grounding: "real \\ud800 mandate"\n'),
     "personas[2].grounding: a character a command line cannot carry"),
    ("X6b", _entry('  - name: Critic\n    grounding: "\\udfff"\n'),
     "personas[2].grounding: a character a command line cannot carry"),
    ("S13", "personas:\n  - name: Auditor\n    tools: [Read]\n"
            f"{_GOOD}  - name: Auditor\n    tools: [Grep]\n",
     "personas[2].name: duplicates personas[0].name"),
]
# The grounding rows are written once with the type each one carries.
_GROUNDING_KIND = {"G1": "int", "G2": "list", "G3": "bool", "G4": "dict", "G5": "date"}
_ILL_TYPED = [(i, text, detail if detail is not None else
               f"personas[1].grounding: {_GROUNDING_KIND[i]}, not a string")
              for i, text, detail in _ILL_TYPED]


def _implicit(tmp_path, text):
    (tmp_path / "panel.yaml").write_text(text, encoding="utf-8")
    return load_personas(tmp_path)


def _explicit(tmp_path, text):
    path = tmp_path / "x.yaml"
    path.write_text(text, encoding="utf-8")
    return load_personas(tmp_path, panel_path=str(path))


# --- rejected versus sourced --------------------------------------------------------

@pytest.mark.parametrize(("row", "text", "detail"), _ILL_TYPED,
                         ids=[r[0] for r in _ILL_TYPED])
def test_an_ill_typed_implicit_entry_discards_the_file_with_its_coordinate(
        tmp_path, row, text, detail):
    """REGRESSION for #115: each row tracebacked, ran silently, or blamed the spawn.

    The whole file is discarded, not the one entry (ruling 2): keeping the good entries
    would source a roster other than the one declared, which is #16's silent shape.
    """
    personas, source, _ = _implicit(tmp_path, text)

    assert source.label == "default"
    assert [(d.origin, d.reason, d.detail) for d in source.discarded] == [
        ("panel.yaml", "malformed", detail)]
    assert "Analyst" not in [p.name for p in personas], "no entry of it survives"


@pytest.mark.parametrize(("row", "text", "detail"), _ILL_TYPED,
                         ids=[r[0] for r in _ILL_TYPED])
def test_an_ill_typed_explicit_entry_is_refused_with_its_coordinate(
        tmp_path, row, text, detail):
    """REGRESSION for #115: ``--panel`` refuses through ``PanelError``, the existing path."""
    with pytest.raises(PanelError) as caught:
        _explicit(tmp_path, text)

    assert str(caught.value).endswith(
        f": it could not be read or parsed ({detail})")


_EVERY_FIELD = """\
personas:
  - name: Analyst
    grounding: Does it compute the right thing?
    tools: [Read, Grep]
    model: some-model
  - name: Sentinel
    grounding:
    tools: []
    model:
  - name: Critic, completeness
    grounding: Find what everyone else missed.
"""


def test_a_well_typed_file_with_every_field_set_still_sources(tmp_path):
    """PAIR, green on main: the contract must not refuse a file it describes.

    Every accepted shape at once: an explicit grant, ``tools: []`` (an explicit grant
    of none, which is not the default), a null grounding (#103's no mandate), a null
    model (the default), an absent ``tools`` (the default grant), and a comma in a
    name, which ruling 3 allows.
    """
    personas, source, _ = _implicit(tmp_path, _EVERY_FIELD)

    assert source.label == "panel file"
    assert source.discarded == ()
    by_name = {p.name: p for p in personas}
    assert by_name["Analyst"].tools == ["Read", "Grep"]
    assert by_name["Analyst"].model == "some-model"
    assert by_name["Sentinel"].tools == []
    assert by_name["Sentinel"].grounding == ""
    assert by_name["Sentinel"].model is None
    assert by_name["Critic, completeness"].tools == list(DEFAULT_ALLOWED_TOOLS)
    assert by_name["Critic, completeness"].model is None


def test_a_well_typed_explicit_file_is_not_refused(tmp_path):
    """PAIR, green on main: the explicit tier sources the same file."""
    personas, source, _ = _explicit(tmp_path, _EVERY_FIELD)

    assert source.label == "--panel"
    assert [p.name for p in personas][:3] == [
        "Analyst", "Sentinel", "Critic, completeness"]


def test_text_argv_can_carry_still_sources(tmp_path):
    """PAIR to X1-X7, green on main: the character rules must not refuse real text.

    A mandate is multi-line prose, so a newline and a tab in ``grounding`` are kept,
    and so are the code points either side of the surrogate block (U+D7FF, U+E000)
    and an astral one, which a range test off by one would refuse. A tool pattern with
    brackets, a colon and a space, and a model id with dots, are printable.
    """
    personas, source, _ = _implicit(tmp_path, (
        "personas:\n  - name: Analyst\n"
        '    grounding: "Line one.\\n\\tLine two: caf\\u00e9 \\ud7ff\\ue000 \\U0001F600"\n'
        '    tools: ["Bash(git diff:*)", Read]\n'
        "    model: some-model-1.5\n"))

    assert source.discarded == ()
    analyst = personas[0]
    assert analyst.grounding == "Line one.\n\tLine two: caf\u00e9 \ud7ff\ue000 \U0001F600"
    assert analyst.tools == ["Bash(git diff:*)", "Read"]
    assert analyst.model == "some-model-1.5"


def test_two_distinct_names_with_distinct_grants_each_keep_their_own(tmp_path):
    """PAIR to S13, green on main: uniqueness must not refuse two different names.

    The invariant is asserted on this side too, not only by comparison with S13: a
    mutation that dropped every grant alike would pass a test that only compared them.
    """
    personas, source, _ = _implicit(tmp_path, (
        "personas:\n  - name: Auditor\n    tools: [Read]\n"
        "  - name: Adversary\n    tools: [Grep]\n    model: m2\n"))

    assert source.discarded == ()
    assert [(p.name, p.tools, p.model) for p in personas][:2] == [
        ("Auditor", ["Read"], None), ("Adversary", ["Grep"], "m2")]


# --- structural shapes: malformed before, malformed now, and said why ---------------

_STRUCTURAL = [
    ("S1", "personas: {name: A}\n", "personas: dict, not a list"),
    ("S2", "personas:\n", "personas: null, not a list"),
    ("S3", "personas: [a, b]\n", "personas[0]: str, not a mapping"),
    ("S4", "- a\n- b\n", "top level: list, not a mapping"),
    ("S5", "just a scalar\n", "top level: str, not a mapping"),
    ("S14", "personas:\n  -\n", "personas[0]: null, not a mapping"),
    ("S8", f"personas:\n{_GOOD}  - grounding: no name here\n", "personas[1].name: missing"),
    ("S9", "personas: " + "[" * 5000 + "]" * 5000 + "\n", "RecursionError"),
    ("S10", "personas:\n\t- name: A\n", "ScannerError at line 2, column 1"),
    ("S11", "personas: !!python/object:os.system\n",
     "ConstructorError at line 1, column 11"),
    ("S15", "[]\n", "top level: list, not a mapping"),
]


@pytest.mark.parametrize(("row", "text", "detail"), _STRUCTURAL,
                         ids=[r[0] for r in _STRUCTURAL])
def test_a_structural_shape_stays_malformed_and_says_where(tmp_path, row, text, detail):
    """Each was ``malformed`` on main ONLY because of the broad ``except Exception``.

    S1–S5, S8 and S14 are green on main in their REASON and red in their DETAIL, which
    was a bare exception type: each raised inside main's broad catch, and the
    validator now names it instead. S9 (pyyaml's composer raising ``RecursionError``,
    #59's trap in ``json.loads``), S10 and S11 are GREEN on main: they come from the
    parse, whose catch stays broad, and they guard it against being narrowed again.

    S15 (a top level of ``[]``) was ``empty`` on main, through ``or {}``: an empty list
    is falsy. Only a null top level, which is what an empty or comment-only file parses
    to, now means "nothing declared" — an empty explicit value is not an absent one
    (#106).
    """
    _, source, _ = _implicit(tmp_path, text)

    assert [(d.reason, d.detail) for d in source.discarded] == [("malformed", detail)]


@pytest.mark.parametrize("text", ["", "# only a comment\n", "personas: []\n",
                                  "colour: red\n"],
                         ids=["S6-empty-file", "comment-only", "S7-empty-list",
                              "no-personas-key"])
def test_a_file_that_declares_nobody_is_still_empty(tmp_path, text):
    """PAIR, green on main: ``empty`` must not become ``malformed``."""
    _, source, _ = _implicit(tmp_path, text)

    assert [(d.reason, d.detail) for d in source.discarded] == [("empty", "")]


def test_an_unknown_key_is_still_ignored(tmp_path):
    """S12, green on main: the contract types the keys it reads and refuses nothing else."""
    personas, source, _ = _implicit(
        tmp_path, "personas:\n  - name: A\n    grounding: g\n    colour: red\n")

    assert source.discarded == ()
    assert personas[0].name == "A"


_PARSER_FAULTS = [
    ("date-out-of-range", "junk: 2024-02-30\n", "ValueError"),
    ("month-out-of-range", "personas:\n  - name: A\n    model: 2024-13-01\n", "ValueError"),
    ("int-past-digit-limit", "junk: " + "9" * 5000 + "\n", "ValueError"),
    ("bad-int-tag", "junk: !!int abc\n", "ValueError"),
    ("bad-bool-tag", "junk: !!bool abc\n", "KeyError"),
    ("bad-timestamp-tag", "junk: !!timestamp abc\n", "AttributeError"),
    # Found by the second pre-merge pass: outside every list the first draft could
    # have enumerated, which is why the catch enumerates nothing.
    ("sexagesimal-float-overflow", "junk: " + "1:" * 3000 + "1.5\n", "OverflowError"),
]


@pytest.mark.parametrize(("text", "detail"), [r[1:] for r in _PARSER_FAULTS],
                         ids=[r[0] for r in _PARSER_FAULTS])
def test_a_fault_inside_the_parser_is_malformed_whatever_it_raises(
        tmp_path, text, detail):
    """GREEN on main, and RED on this fix's first draft: it guards the parse's catch.

    pyyaml does not raise only ``YAMLError``. Its constructors raise a bare
    ``ValueError``, ``KeyError`` or ``AttributeError`` for a scalar they cannot build,
    anywhere in the file — even under a key nothing reads. The first draft narrowed
    the parse's catch to ``(yaml.YAMLError, RecursionError)``, and every row here
    became a traceback on both tiers; the pre-merge panel found it by measuring, and
    the stub tests that should have caught it had been rewritten to raise a fake
    ``YAMLError``. The parse is the library's, handed only the operator's text, so
    whatever it raises is the file's fault.
    """
    _, source, _ = _implicit(tmp_path, text)
    with pytest.raises(PanelError) as caught:
        _explicit(tmp_path, text)

    assert [(d.reason, d.detail) for d in source.discarded] == [("malformed", detail)]
    assert str(caught.value).endswith(f"({detail})")


# --- closed by default --------------------------------------------------------------

@pytest.mark.parametrize("where", ["_declared_entries", "_declared_persona", "_unique"])
@pytest.mark.parametrize("raised", [RuntimeError, ValueError, TypeError, KeyError])
def test_a_bug_in_the_validation_stays_loud(tmp_path, monkeypatch, where, raised):
    """A fault the validator does not NAME is a bug, and a bug escapes (#59, #51).

    Not reachable on main, whose validation was none: its ``except Exception`` wrapped
    the parse and the build together and recorded every fault as the operator's
    malformed file, with the traceback that would have found it gone. The validation
    is OURS, so its catch names only ``_DeclarationError``. ``ValueError`` kills the
    rejected catch of that error's base class, and ``TypeError`` and ``KeyError`` a
    catch of what main's faults happened to raise. All three functions sharing the
    catch are covered, not only the one most likely to hold a bug.
    """
    def _broken(*_a, **_k):
        raise raised("a bug, not the operator's file")

    monkeypatch.setattr(personas_module, where, _broken)

    with pytest.raises(raised):
        _implicit(tmp_path, "personas:\n  - name: A\n")


# --- the detail never carries the value ---------------------------------------------

_MARKER = "INTERNAL-ONLY acme-billing PROJ-4471"


@pytest.mark.parametrize("text", [
    f'personas:\n  - name: "{_MARKER}\\t"\n',
    f"personas:\n  - name: {_MARKER}\n    grounding: [{_MARKER}]\n",
    f"personas:\n  - name: {_MARKER}\n    tools: {_MARKER}\n",
    f"personas:\n  - name: {_MARKER}\n    tools: [{_MARKER}, 1]\n",
    f"personas:\n  - name: {_MARKER}\n    model: [{_MARKER}]\n",
    f"personas:\n  - name: {_MARKER}\n  - name: {_MARKER}\n",
    f"personas:\n  - {_MARKER}\n",
    f"{_MARKER}\n",
], ids=["name", "grounding", "tools", "tools-item", "model", "duplicate", "entry",
        "top-level"])
def test_a_detail_names_a_coordinate_never_the_value(tmp_path, text):
    """#105's rule: a record carries names and coordinates, never the content.

    A ``name`` is operator text, and the duplicate rule is the one most tempted to
    quote it. Each row puts a marker in the field at fault and asserts it is absent
    from both the discard and the refusal. The ``entry`` and ``top-level`` rows are
    GREEN on main, where the detail was a bare ``AttributeError``: they guard the new
    structural details, which are the likeliest to be tempted into quoting.
    """
    _, source, _ = _implicit(tmp_path, text)
    with pytest.raises(PanelError) as caught:
        _explicit(tmp_path, text)

    assert source.discarded[0].reason == "malformed"
    assert _MARKER not in source.discarded[0].detail
    assert "INTERNAL" not in str(caught.value)


# --- the other tiers ----------------------------------------------------------------

def test_the_markdown_tiers_can_only_yield_strings():
    """The tiers this contract does not cover, asserted rather than assumed (#103).

    ``parse_claude_md_experts`` builds every field from regex groups over text, so it
    cannot produce a non-string ``name`` or ``grounding``, and it sets neither ``tools``
    nor ``model``: they keep the dataclass defaults. That is why the validator sits on
    the YAML branch alone. Green on main by construction.
    """
    personas = parse_claude_md_experts(
        "# Resident Experts\n\n## A — Role\n\nBody.\n\n## B — Role\n")

    assert personas
    for p in personas:
        assert type(p.name) is str and type(p.grounding) is str
        assert p.tools == list(DEFAULT_ALLOWED_TOOLS) and p.model is None
