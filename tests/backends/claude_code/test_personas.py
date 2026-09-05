# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Persona sourcing must notice it fell back, not merely label the result (#16).

``load_personas`` returned the single string ``"default"`` for six distinct
causes, three of which produced a byte-identical roster, stdout and artefact. An
operator whose declared panel was discarded saw exactly what an operator who
declared none saw — the first exit criterion's class test (#40, #74), which is
**distinguishability** rather than the presence of a word.

**Sourcing had no dedicated test module before this one.** ``load_personas``,
``parse_claude_md_experts`` and ``_load_panel_file`` were exercised only
incidentally, through CLI tests whose subject was something else. That gap is
inherited rather than created here, and the six causes are its natural first
tests.

The one principle every case falls out of: **a degradation is a persona
declaration that was reached and yielded no personas.** A *declaration* is the
thing whose sole purpose is to declare a panel — ``--panel`` by construction,
``panel.yaml``/``panel.md`` by existing at all, and inside ``CLAUDE.md`` (a file
that exists for other reasons) the ``Resident Experts`` heading. So a repo with
no ``CLAUDE.md``, and a ``CLAUDE.md`` with no experts heading, declared nothing
and must read **clean**: a rule that fires on a healthy run is the mirror image
of the defect it reports.
"""

from __future__ import annotations

import sys
import types

import pytest

from kuang.backends.claude_code.personas import (DISCARD_ORIGINS,
                                                 DISCARD_REASONS, PanelError,
                                                 load_personas)

# A Resident Experts heading the parser commits to, with subsections it cannot
# read: ``###`` instead of ``##``, and a colon instead of the " — " separator.
# Two plausible slips, measured on main as producing a roster, stdout and
# artefact byte-identical to a repo with no CLAUDE.md at all.
_EXPERTS_THAT_PARSE_TO_ZERO = """\
# Resident Experts

### The Analyst: correctness

Does the change compute the right thing?
"""

_GOOD_EXPERTS = """\
# Resident Experts

## Analyst — Correctness

Does the change compute the right thing?
"""

_GOOD_PANEL_MD = """\
# Resident Experts

## Auditor — Constraints

What external rules must this not violate?
"""


def _repo(tmp_path, **files):
    """A bare directory carrying the named files. Sourcing reads files, not git."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def _no_pyyaml(monkeypatch):
    """Force ``import yaml`` to fail, whatever the ambient venv has.

    ``pyyaml`` is an optional extra (``kuang[yaml]``) and is absent from this
    repo's venv, so cause 4 is what a plain ``pip install`` actually does. It is
    forced rather than assumed so the test asserts the branch, not the machine.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)


def _stub_yaml(monkeypatch, payload):
    """A ``yaml`` module whose ``safe_load`` returns ``payload``.

    Cause 5 (a ``panel.yaml`` that parses to zero personas) is **unreachable**
    without a yaml parser: ``import yaml`` raises first, which is cause 4. Checked
    by execution before this test was written (#72's trap, #87's rule), and the
    stub is proved non-vacuous by ``test_the_yaml_stub_is_not_a_no_op`` below.
    """
    stub = types.ModuleType("yaml")
    stub.safe_load = lambda _text: payload
    monkeypatch.setitem(sys.modules, "yaml", stub)


# --- the six causes ----------------------------------------------------------
#
# Every cause returns the label "default" on main. Only the ``discarded`` column
# tells them apart, which is the whole of this issue.

_CAUSES = [
    ({}, None, ()),
    ({"CLAUDE.md": "# A repo\n\nNo panel here.\n"}, None, ()),
    ({"CLAUDE.md": _EXPERTS_THAT_PARSE_TO_ZERO}, None, (("CLAUDE.md", "empty"),)),
    ({"panel.yaml": "personas:\n  - name: A\n"}, "absent",
     (("panel.yaml", "unsupported"),)),
    ({"panel.yaml": "personas: []\n"}, {"personas": []},
     (("panel.yaml", "empty"),)),
    ({"panel.md": "# Notes\n\nNothing declaring a persona.\n"}, None,
     (("panel.md", "empty"),)),
]
_IDS = [
    "1-no-CLAUDE.md-and-no-panel-file",
    "2-CLAUDE.md-with-no-experts-heading",
    "3-experts-heading-parsing-to-zero-personas",
    "4-panel.yaml-with-pyyaml-absent",
    "5-panel.yaml-parsing-to-zero-personas",
    "6-panel.md-parsing-to-zero-personas",
]


@pytest.mark.parametrize(("files", "yaml_state", "expected"), _CAUSES, ids=_IDS)
def test_each_cause_of_the_default_panel_is_recorded_distinctly(
        tmp_path, monkeypatch, files, yaml_state, expected):
    """Six causes, one label — and now six distinguishable records."""
    if yaml_state == "absent":
        _no_pyyaml(monkeypatch)
    elif yaml_state is not None:
        _stub_yaml(monkeypatch, yaml_state)

    _, source, _ = load_personas(_repo(tmp_path, **files))

    assert source.label == "default"
    assert tuple((d.origin, d.reason) for d in source.discarded) == expected


def test_a_discarded_declaration_is_distinguishable_from_none(tmp_path, monkeypatch):
    """The class test (#40, #74): causes 1 and 3 must not read the same.

    On main their roster, stdout and artefact were byte-identical. This is the
    single assertion the issue exists for.
    """
    nothing = tmp_path / "nothing"
    nothing.mkdir()
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "CLAUDE.md").write_text(_EXPERTS_THAT_PARSE_TO_ZERO)

    _, clean, _ = load_personas(nothing)
    _, degraded, _ = load_personas(broken)

    assert clean != degraded
    assert clean.degraded is False
    assert degraded.degraded is True


@pytest.mark.parametrize("files", [
    {},
    {"CLAUDE.md": "# A repo\n\nNo panel here.\n"},
    {"CLAUDE.md": _GOOD_EXPERTS},
], ids=["nothing-declared", "CLAUDE.md-declaring-no-panel", "a-panel-that-worked"])
def test_a_run_that_declared_nothing_reads_clean(tmp_path, files):
    """The mirror image: a rule that fires on a healthy run is the defect again.

    A repo that asked for nothing must carry no discard and no warning at all.
    Paired with the test above, which forbids the vacuous way to pass it.
    """
    _, source, _ = load_personas(_repo(tmp_path, **files))

    assert source.discarded == ()
    assert source.degraded is False


def test_an_unreached_declaration_is_not_recorded(tmp_path, monkeypatch):
    """Only tiers the precedence chain actually consulted can be discarded.

    A good ``CLAUDE.md`` answers at tier 1, so a broken ``panel.yaml`` beside it
    was never asked for and is not a degradation. Recording it would be a claim
    about a file this run never opened.
    """
    _no_pyyaml(monkeypatch)
    repo = _repo(tmp_path, **{"CLAUDE.md": _GOOD_EXPERTS,
                              "panel.yaml": "personas:\n  - name: A\n"})

    _, source, _ = load_personas(repo)

    assert source.label == "CLAUDE.md"
    assert source.discarded == ()


def test_a_discarded_tier_is_recorded_even_when_a_later_tier_succeeds(
        tmp_path, monkeypatch):
    """The roster is not the fallback, and a declaration was still discarded.

    Both facts are true at once, so the record cannot be a single "did we fall
    back to the default" bit.
    """
    _no_pyyaml(monkeypatch)
    repo = _repo(tmp_path, **{"CLAUDE.md": _EXPERTS_THAT_PARSE_TO_ZERO,
                              "panel.yaml": "personas:\n  - name: A\n",
                              "panel.md": _GOOD_PANEL_MD})

    personas, source, _ = load_personas(repo)

    assert source.label == "panel file"
    assert "Auditor" in [p.name for p in personas]
    assert tuple((d.origin, d.reason) for d in source.discarded) == (
        ("CLAUDE.md", "empty"), ("panel.yaml", "unsupported"))


def test_the_missing_optional_extra_is_not_reported_as_a_malformed_file(
        tmp_path, monkeypatch):
    """An environment fault and an operator input fault have different remedies.

    Both landed in one ``except Exception`` on main. ``unsupported`` says install
    the extra; ``malformed`` says fix the file. Collapsing them is this issue's
    own conflation, one field along.
    """
    _no_pyyaml(monkeypatch)
    absent = load_personas(_repo(tmp_path / "a", **{"panel.yaml": "personas: []\n"}))[1]

    _stub_yaml(monkeypatch, None)
    monkeypatch.setattr(sys.modules["yaml"], "safe_load",
                        lambda _t: (_ for _ in ()).throw(ValueError("bad token")))
    broken = load_personas(_repo(tmp_path / "b", **{"panel.yaml": "{{\n"}))[1]

    assert absent.discarded[0].reason == "unsupported"
    assert broken.discarded[0].reason == "malformed"
    assert "bad token" in broken.discarded[0].detail


def test_a_panel_yaml_that_names_nobody_answers_its_tier(tmp_path, monkeypatch):
    """The precedence #16 deliberately did NOT change, pinned so it cannot drift.

    A ``panel.yaml`` that *parses* answers the panel-file tier even when it names
    nobody; only one that could not be parsed at all falls through to
    ``panel.md``. That is what ``main`` did before #16, and #16 is a reporting
    issue — what a run *does* is untouched, and only what it *says* is new.

    Reachable only with a YAML parser present (without one the branch is
    ``unsupported``, which does fall through), so it is asserted with the stub
    rather than against the ambient venv.
    """
    _stub_yaml(monkeypatch, {"personas": []})
    repo = _repo(tmp_path, **{"panel.yaml": "personas: []\n",
                              "panel.md": _GOOD_PANEL_MD})

    personas, source, _ = load_personas(repo)

    assert source.label == "default"
    assert "Auditor" not in [p.name for p in personas]
    assert tuple((d.origin, d.reason) for d in source.discarded) == (
        ("panel.yaml", "empty"),)


def test_the_yaml_stub_is_not_a_no_op(tmp_path, monkeypatch):
    """A 0-red mutation is a claim, not a result (#87) — and so is a stub.

    If ``_stub_yaml`` could not load a panel either, cause 5 would be passing for
    the same reason cause 4 does and would prove nothing.
    """
    _stub_yaml(monkeypatch, {"personas": [{"name": "A", "grounding": "g"}]})

    personas, source, _ = load_personas(
        _repo(tmp_path, **{"panel.yaml": "personas:\n  - name: A\n"}))

    assert source.label == "panel file"
    assert source.discarded == ()
    assert "A" in [p.name for p in personas]


# --- the closed sets ---------------------------------------------------------

def test_origin_and_reason_are_categorical(tmp_path, monkeypatch):
    """Control values are set at the source, never parsed from prose.

    There is no model anywhere in this path, which makes it the easiest place in
    the codebase to be exact. Every value a run can emit must be in the declared
    set, or ``check_invariants.py`` and any consumer are reading an open vocabulary.
    """
    _no_pyyaml(monkeypatch)
    seen = []
    for index, (files, yaml_state, _) in enumerate(_CAUSES):
        if yaml_state not in (None, "absent"):
            _stub_yaml(monkeypatch, yaml_state)
        else:
            _no_pyyaml(monkeypatch)
        seen.extend(load_personas(_repo(tmp_path / str(index), **files))[1].discarded)

    assert seen, "the causes must actually produce discards, or this is vacuous"
    assert {d.origin for d in seen} <= set(DISCARD_ORIGINS)
    assert {d.reason for d in seen} <= set(DISCARD_REASONS)


def test_load_personas_still_returns_a_three_tuple(tmp_path):
    """``probe.py`` indexes ``load_personas(repo)[0]``; the arity is a contract.

    The source label became a record rather than a fourth element for exactly
    this reason.
    """
    result = load_personas(tmp_path)

    assert len(result) == 3
    assert [p.name for p in result[0]]


# --- --panel: an explicit declaration ----------------------------------------

def test_panel_path_takes_precedence_over_every_other_tier(tmp_path, monkeypatch):
    """An operator who names a file gets that file, and nothing else is consulted."""
    _no_pyyaml(monkeypatch)
    named = tmp_path / "docs" / "experts.md"
    named.parent.mkdir()
    named.write_text(_GOOD_PANEL_MD)
    repo = _repo(tmp_path, **{"CLAUDE.md": _GOOD_EXPERTS,
                              "panel.yaml": "personas:\n  - name: A\n",
                              "panel.md": _EXPERTS_THAT_PARSE_TO_ZERO})

    personas, source, _ = load_personas(repo, panel_path=named)

    assert source.label == "--panel"
    assert source.path == str(named)
    assert source.discarded == ()
    assert "Auditor" in [p.name for p in personas]
    assert "Analyst" not in [p.name for p in personas]


def test_panel_path_accepts_yaml(tmp_path, monkeypatch):
    """Dispatch is by suffix, reusing both loaders rather than adding a third."""
    _stub_yaml(monkeypatch, {"personas": [{"name": "Named", "grounding": "g"}]})
    named = tmp_path / "team.yaml"
    named.write_text("personas:\n  - name: Named\n")

    personas, source, _ = load_personas(tmp_path, panel_path=named)

    assert source.label == "--panel"
    assert "Named" in [p.name for p in personas]


@pytest.mark.parametrize(("filename", "text", "yaml_state", "expected"), [
    ("missing.md", None, None, "does not exist"),
    ("empty.md", "# Notes\n\nNothing here.\n", None, "no persona"),
    ("panel.yaml", "personas: []\n", {"personas": []}, "no persona"),
    ("panel.yaml", "{{\n", "absent", "kuang[yaml]"),
], ids=["missing-file", "markdown-naming-nobody", "yaml-naming-nobody",
        "yaml-without-the-optional-extra"])
def test_an_explicit_panel_that_cannot_be_used_refuses(
        tmp_path, monkeypatch, filename, text, yaml_state, expected):
    """An explicit instruction is never silently overridden.

    Falling back here would be this issue's own defect one tier up, with the
    operator having been maximally explicit — so ``--panel`` refuses where the
    implicit tiers record and continue. What the resulting exit code MEANS is
    #32's; this only declines to substitute a panel nobody asked for.
    """
    if yaml_state == "absent":
        _no_pyyaml(monkeypatch)
    elif yaml_state is not None:
        _stub_yaml(monkeypatch, yaml_state)
    named = tmp_path / filename
    if text is not None:
        named.write_text(text)

    with pytest.raises(PanelError) as exc:
        load_personas(tmp_path, panel_path=named)

    assert filename in str(exc.value)
    assert expected in str(exc.value)
