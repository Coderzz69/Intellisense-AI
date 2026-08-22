"""Puts the canonical AI package on ``sys.path`` for the scripts in this bundle.

The ``traffic_pipeline`` package used to be duplicated inside this folder and had
drifted out of sync with the copy under ``traffic/``. There is now one copy, and
these training/dataset scripts reach it through here.
"""

from __future__ import annotations

import sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
AI_ROOT = BUNDLE_ROOT.parent / "traffic"

if not (AI_ROOT / "traffic_pipeline" / "__init__.py").exists():
    raise ImportError(
        f"Canonical AI package not found at {AI_ROOT}. "
        "It should live in the project's 'traffic/' directory."
    )

for path in (str(BUNDLE_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
