# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Agilit Ltd
"""``python -m kuang`` — the same entry point as the ``kuang`` script."""

from kuang.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
