# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The subprocess boundary: finding the ``claude`` binary and calling it.

Everything that crosses out of this process goes through here, so the permission
flags a reviewer runs under are visible in one place.
"""

from __future__ import annotations

import os
import shutil


def _resolve_claude_bin() -> str:
    """Locate the ``claude`` executable robustly (child PATH may differ).

    Order: ``$CLAUDE_BIN`` -> ``PATH`` -> ``~/.local/bin/claude``. Falls back to
    the bare name so the failure, if any, is a clear FileNotFoundError.
    """
    for cand in (os.environ.get("CLAUDE_BIN"), shutil.which("claude"),
                 os.path.expanduser("~/.local/bin/claude")):
        if cand and os.path.exists(cand):
            return cand
    return "claude"
