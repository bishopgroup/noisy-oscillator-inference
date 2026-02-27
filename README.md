# Noisy Oscillator Inference

Tools for Bayesian/EM-style inference in noisy oscillator data.

Current capabilities (in this repository):
- infer hidden phase trajectories
- estimate oscillation frequency
- estimate phase diffusivity
- estimate measurement noise magnitude

This repository currently contains a synthetic linear-measurement example, and is
structured to expand to additional oscillator models and datasets.

## Current Example

Main files:
- `src/noisy_oscillator/synthetic_phase_linear.py` (shared core implementation)
- `examples/synthetic_phase_linear/SyntheticPhaseLinearMeasurement.py` (compatibility wrapper for notebooks)
- `examples/synthetic_phase_linear/SyntheticPhaseLinearMeasurement.ipynb`
- `examples/synthetic_phase_linear/SPLM_Performance.ipynb`

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
pip install jupyter
```

3. Run the notebooks:

```bash
jupyter lab
```

Then open:
- `examples/synthetic_phase_linear/SyntheticPhaseLinearMeasurement.ipynb` for method walkthrough
- `examples/synthetic_phase_linear/SPLM_Performance.ipynb` for benchmark/sensitivity sweeps

## Collaboration Workflow

Suggested workflow for contributors:
1. Create a feature branch from `main`.
2. Keep model logic in `.py` files and use notebooks for demos/figures.
3. Commit small, focused changes with clear messages.
4. Open a pull request for review before merge.

## Repository Structure

```text
examples/
  synthetic_phase_linear/
  <other_examples>/
src/
  noisy_oscillator/
```

As additional examples are added, each should live in its own folder under
`examples/` with:
- model code (`.py`)
- demonstration notebooks (`.ipynb`)
- associated generated artifacts (`.pdf`, `.csv`) when needed

Reusable logic shared across examples should be placed in `src/noisy_oscillator/`.
