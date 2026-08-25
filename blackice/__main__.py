# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""``python -m blackice`` — the same entry point as the ``blackice`` script."""

from blackice.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
