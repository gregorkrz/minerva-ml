"""Shim: implementation is in ``src.eval.classification_plots``.

Keeps ``from eval_classification_plots import …`` working when ``notebooks/``
is on ``sys.path`` (typical notebook setups).
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.eval.classification_plots import *  # noqa: F403
