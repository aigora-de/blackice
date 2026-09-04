# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""The Claude Code backend: one ``claude -p`` subprocess per persona per epoch.

Six jobs, six modules — persona sourcing, surface assembly, the subprocess
boundary, the findings contract, memory, and the semantic clusterer — plus the
permission policy they share and the ``session`` that wires them into the
engine's seams. This is the backend's public surface.
"""

from .cluster import ReduceState
from .contract import UNRESOLVED_SEVERITY
from .memory import load_prior_findings, ungrounded_keys
from .permissions import (DEFAULT_ALLOWED_TOOLS, DEFAULT_DISALLOWED_TOOLS,
                          unavailable_tools)
from .personas import (COMPLETENESS_CRITIC, DEFAULT_PERSONAS, REQUIRED_LENSES,
                       SURVIVABILITY, LensCoverage, Persona, load_personas,
                       parse_claude_md_experts)
from .session import PanelSession
from .spawn import called_no_tool
from .surface import (SurfaceError, SurfaceRecord, build_path_surface,
                      gather_diff)

__all__ = [
    "COMPLETENESS_CRITIC", "DEFAULT_ALLOWED_TOOLS", "DEFAULT_DISALLOWED_TOOLS",
    "DEFAULT_PERSONAS", "LensCoverage", "PanelSession", "Persona", "ReduceState",
    "REQUIRED_LENSES", "SURVIVABILITY", "SurfaceError", "SurfaceRecord",
    "UNRESOLVED_SEVERITY", "build_path_surface", "called_no_tool",
    "gather_diff",
    "load_personas", "load_prior_findings", "parse_claude_md_experts",
    "unavailable_tools", "ungrounded_keys",
]
