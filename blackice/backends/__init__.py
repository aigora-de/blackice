# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""Backends: bindings from the engine's seams to a particular agent runtime.

A different runtime is a different subpackage. Nothing here is imported by
``blackice.engine`` — the dependency runs one way only, and a test enforces it.
"""
