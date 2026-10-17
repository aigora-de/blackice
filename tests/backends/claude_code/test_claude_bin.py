# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Finding the ``claude`` binary: a usable one, or a named refusal (#51).

``resolve_claude_bin`` replaces a resolver that tested ``if cand and
os.path.exists(cand)`` and fell back to the bare name ``claude``. That admitted a
directory and a non-executable file, each of which then failed at the first spawn.
It also SKIPPED two values silently, falling through to ``PATH`` and
``~/.local/bin/claude``: a ``$CLAUDE_BIN`` naming a path that does not exist, and an
empty ``$CLAUDE_BIN`` (read by truthiness, #106's defect). So an operator who pinned
a binary and mistyped it got a different one, and the run spent on it. Both were
ruled members of the first-criterion class and folded into #51.

Pinned as pairs, because each half passes trivially without the other:

* nothing usable is refused, naming ``$CLAUDE_BIN``, ``PATH`` and
  ``~/.local/bin/claude`` — PAIRED WITH each tier, alone, being found, and ``PATH``
  winning over ``~/.local/bin/claude`` when both are there;
* an explicit ``$CLAUDE_BIN`` that cannot be used is refused, and never searched
  past, even with a working binary in both later tiers — PAIRED WITH an explicit
  usable one being the binary returned;
* a directory, a non-executable file, a FIFO and a symlink loop are refused —
  PAIRED WITH a symlink to an executable, the shape of the installer's
  ``~/.local/bin/claude``, passing;
* faults the resolver does not name stay loud: the refusal is closed by default.

Hermetic by construction, not by luck: every test sets ``HOME``, ``PATH`` and
``CLAUDE_BIN`` itself, and ``PATH`` is a directory this test made. A mutant that
falls through cannot reach the developer's real ``claude``.

**Which are regressions, measured on main with this file in place** and a shim for
the new names. Recorded in the PR body; any test green on main says so in its own
docstring.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from kuang.backends.claude_code import ClaudeBinError
from kuang.backends.claude_code import spawn as spawn_module
from kuang.backends.claude_code.spawn import resolve_claude_bin


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 97\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An environment in which nothing resolves until a test puts something there."""
    home, path = tmp_path / "home", tmp_path / "pathdir"
    home.mkdir()
    path.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(path))
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.chdir(tmp_path)
    return {"home": home, "path": path, "local": home / ".local" / "bin" / "claude",
            "root": tmp_path}


def _refusal() -> str:
    with pytest.raises(ClaudeBinError) as info:
        resolve_claude_bin()
    return str(info.value)


# --- nothing usable, and each tier found ---------------------------------------

def test_nothing_usable_is_refused_naming_every_place_looked(env):
    message = _refusal()
    assert message.startswith("claude: no usable binary was found")
    for place in ("$CLAUDE_BIN is unset", "no `claude` on PATH",
                  "~/.local/bin/claude does not exist"):
        assert place in message
    # In search order, so the message reads as the search did.
    assert (message.index("$CLAUDE_BIN") < message.index("PATH")
            < message.index("~/.local/bin/claude"))
    # Written the short way: a home directory is a person's name on most installs.
    assert str(env["home"]) not in message


def test_the_refusal_says_what_to_do(env):
    assert "install the Claude Code CLI, or set $CLAUDE_BIN to its path" in _refusal()


def test_claude_on_path_is_found(env):
    """PAIR, green on main: the refusal does not fire when ``PATH`` holds one."""
    found = _exe(env["path"] / "claude")
    assert resolve_claude_bin() == str(found)


def test_the_installer_symlink_is_found(env):
    """PAIR, green on main: the installer's shape, a symlink to a versioned file."""
    target = _exe(env["home"] / ".local" / "share" / "claude" / "versions" / "9.9.9")
    env["local"].parent.mkdir(parents=True)
    env["local"].symlink_to(target)
    assert resolve_claude_bin() == str(env["local"])


def test_path_is_searched_before_the_local_install(env):
    """PAIR, green on main: the search order has not moved."""
    _exe(env["local"])
    on_path = _exe(env["path"] / "claude")
    assert resolve_claude_bin() == str(on_path)


@pytest.mark.parametrize("shape, said", [
    ("dir", "~/.local/bin/claude is a directory, not an executable file"),
    ("noexec", "~/.local/bin/claude is not executable"),
])
def test_an_unusable_local_install_is_named(env, shape, said):
    if shape == "dir":
        env["local"].mkdir(parents=True)
    else:
        _exe(env["local"]).chmod(0o644)
    message = _refusal()
    assert said in message
    assert str(env["home"]) not in message


def test_a_relative_path_entry_is_returned_absolute(env, monkeypatch):
    """What is checked is what is spawned: the spawn runs with ``cwd`` the repository.

    A relative result from a relative ``PATH`` entry would be looked up again,
    relative to the repository, by ``subprocess.run`` — a different file, or none.
    """
    _exe(env["root"] / "rel" / "claude")
    monkeypatch.setenv("PATH", "rel")
    assert resolve_claude_bin() == str(env["root"] / "rel" / "claude")


# --- an explicit $CLAUDE_BIN ---------------------------------------------------

@pytest.fixture
def both_tiers_work(env):
    """A working ``claude`` in BOTH later tiers, so a fallthrough would succeed."""
    _exe(env["path"] / "claude")
    _exe(env["local"])
    return env


def test_an_explicit_usable_binary_is_the_one_returned(both_tiers_work, monkeypatch):
    """PAIR, green on main: an explicit binary still wins over both later tiers."""
    pinned = _exe(both_tiers_work["root"] / "pinned" / "claude")
    monkeypatch.setenv("CLAUDE_BIN", str(pinned))
    assert resolve_claude_bin() == str(pinned)


def test_an_explicit_symlink_passes(both_tiers_work, monkeypatch):
    """PAIR, green on main: the usability check follows a symlink."""
    target = _exe(both_tiers_work["root"] / "versions" / "9.9.9")
    link = both_tiers_work["root"] / "link"
    link.symlink_to(target)
    monkeypatch.setenv("CLAUDE_BIN", str(link))
    assert resolve_claude_bin() == str(link)


def test_a_relative_explicit_binary_is_returned_absolute(both_tiers_work, monkeypatch):
    """Resolved against the directory it was checked in, not the repository's."""
    _exe(both_tiers_work["root"] / "pinned" / "claude")
    monkeypatch.setenv("CLAUDE_BIN", "pinned/claude")
    assert resolve_claude_bin() == str(both_tiers_work["root"] / "pinned" / "claude")


def test_an_explicit_missing_binary_is_refused_not_searched_past(both_tiers_work,
                                                                  monkeypatch):
    """REGRESSION: on main this resolved to the ``PATH`` binary, silently."""
    monkeypatch.setenv("CLAUDE_BIN", "/nonexistent/claude")
    message = _refusal()
    assert message.startswith("$CLAUDE_BIN /nonexistent/claude: does not exist")
    assert "is not replaced by a search of PATH or ~/.local/bin/claude" in message


def test_an_empty_explicit_binary_is_refused(both_tiers_work, monkeypatch):
    """REGRESSION: on main ``""`` was read by truthiness and the search ran (#106)."""
    monkeypatch.setenv("CLAUDE_BIN", "")
    message = _refusal()
    assert message.startswith("$CLAUDE_BIN: set but empty")
    assert "unset it to search PATH and then ~/.local/bin/claude" in message


@pytest.mark.parametrize("value", [" ", "\t"])
def test_a_blank_explicit_binary_is_refused_as_missing(both_tiers_work, monkeypatch,
                                                       value):
    """Not "empty": it names a file, ``" "``, that does not exist, and says so."""
    monkeypatch.setenv("CLAUDE_BIN", value)
    assert f"$CLAUDE_BIN {value}: does not exist" in _refusal()


def test_an_explicit_directory_is_refused(both_tiers_work, monkeypatch):
    monkeypatch.setenv("CLAUDE_BIN", str(both_tiers_work["root"]))
    assert ": is a directory, not an executable file" in _refusal()


def test_an_explicit_non_executable_file_is_refused(both_tiers_work, monkeypatch):
    plain = both_tiers_work["root"] / "plain"
    plain.write_text("x\n")
    plain.chmod(0o644)
    monkeypatch.setenv("CLAUDE_BIN", str(plain))
    assert f"$CLAUDE_BIN {plain}: is not executable" in _refusal()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_an_explicit_fifo_is_refused(both_tiers_work, monkeypatch):
    fifo = both_tiers_work["root"] / "fifo"
    os.mkfifo(fifo, 0o755)
    monkeypatch.setenv("CLAUDE_BIN", str(fifo))
    assert f"$CLAUDE_BIN {fifo}: is not a regular file" in _refusal()


def test_an_explicit_symlink_loop_is_refused_by_errno(both_tiers_work, monkeypatch):
    loop = both_tiers_work["root"] / "loop"
    loop.symlink_to(loop)
    monkeypatch.setenv("CLAUDE_BIN", str(loop))
    assert (f"$CLAUDE_BIN {loop}: OSError [errno {errno.ELOOP}]: "
            f"{os.strerror(errno.ELOOP)}") in _refusal()


# --- closed by default ---------------------------------------------------------

@pytest.mark.parametrize("fault", [ValueError("a bug"), PermissionError(13, "a bug")])
def test_a_fault_the_resolver_does_not_name_stays_loud(env, monkeypatch, fault):
    """Green on main, which caught nothing: guards the REJECTED wider catch.

    Only the faults it can name become a refusal; a bug is never "your binary".

    ``PermissionError`` is an ``OSError``, so this is also red against the rejected
    design of turning every ``OSError`` into an operator error.
    """
    def _broken(*_a, **_k):
        raise fault
    monkeypatch.setattr(spawn_module.shutil, "which", _broken)
    with pytest.raises(type(fault)) as info:
        resolve_claude_bin()
    assert not isinstance(info.value, ClaudeBinError)
