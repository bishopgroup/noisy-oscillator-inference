"""Compatibility wrapper for legacy notebook imports.

Notebooks in this example historically imported:
    import SyntheticPhaseLinearMeasurement as SPLM

Core logic now lives in:
    src/noisy_oscillator/synthetic_phase_linear.py
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from noisy_oscillator.synthetic_phase_linear import *  # noqa: F401,F403
