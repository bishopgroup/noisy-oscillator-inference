# Examples Directory

Each subfolder under `examples/` should contain one self-contained use case.

Recommended contents per example:
- `<example_name>.py` or similarly named module with core methods
- one or more notebooks for walkthrough and benchmarks
- generated figure/data artifacts only if they are part of the reproducible result

Suggested naming:
- `synthetic_phase_linear`
- `synthetic_phase_nonlinear`
- `experimental_<dataset_or_system>`

Contributor checklist:
1. Add a short section in root `README.md` linking to the new example.
2. Put reusable logic in `src/noisy_oscillator/`; keep notebooks focused on explanation and plots.
3. Verify notebooks run top-to-bottom from a clean kernel.
