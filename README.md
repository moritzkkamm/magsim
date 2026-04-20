# gs_generator

Batched Gray-Scott reaction-diffusion generator for synthesizing
magnetic-domain-like 2D patterns. Intended as a source of plausible
ground-truth domain configurations for ML training.

## What it does

Simulates the Gray-Scott PDE

    dA/dt = D_A * lap(A) - A*B^2 + f*(1 - A)
    dB/dt = D_B * lap(B) + A*B^2 - (k + f)*B

on a 2D grid with periodic boundaries. Returns continuous (A, B) fields
plus per-sample metadata (f, k, theta, alpha, stop_step).

- CuPy on GPU preferred, NumPy fallback for debugging.
- Batched: N independent grids evolved in parallel.
- `generate(...)` is the high-level entry point.

## Current features

- **Core Gray-Scott**: D_A, D_B, f, k, dt, Sims 3x3 Laplacian stencil.
- **Region-based (f, k) sampling**: named morphology classes
  (`labyrinth`, `bubbles`, `stripes`, `coral`, `mitosis`, `solitons`,
  `holes`) drawn from hand-tuned boxes in (f, k) space.
- **Spatially-varying (f, k)**: low-frequency Gaussian random fields
  added on top of mean (f, k) — mimics sample inhomogeneity. Works well.
- **Anisotropic diffusion**: per-sample theta, alpha. Stable but the
  orientation effect is weak (see bugs).
- **Randomized seeding**: number, position, and radius of initial B patches.
- **Random-stop**: per-sample stop step for diverse transient states.
- **`meta` dict**: all generating parameters returned alongside the fields.

## Quick start

```python
from gs_generator import generate
A, B, meta = generate(batch=16, H=256, W=256, n_steps=8000,
                      region="labyrinth", random_stop=True, seed=0)
# B: (16, 256, 256) float32 in [0, 1]
```

See `quick_check.py` for a runnable demo.

## Known bugs / limitations

- **Anisotropic mode is weak**: directional bias is subtle and occasionally
  produces ring artifacts. Root cause is that the isotropic Sims stencil is
  tightly calibrated for dt=1, D=1, so the tensor correction has to stay
  small. Proper fix needs dt < 1 and a rebalanced tensor stencil.
- **Ring artifacts in random-stop**: some early-stopped samples with few
  seeds are single propagating wavefronts (bullseyes) that haven't broken
  up into labyrinths yet. Mitigation: raise `n_seeds_range` to e.g.
  `(15, 30)` in the config.
- **Occasional collapse near region boundaries**: `MORPHOLOGY_REGIONS`
  boxes were hand-tuned, not measured. A few (f, k) draws still fall into
  trivial regimes. Tightening is ad hoc.
- **No convergence check**: runs exactly `n_steps`, even if B has settled.
- **Gray-Scott is not micromagnetics**: patterns have the right morphology
  class for ML training but do not encode real domain-wall physics. Do not
  use these as diffmag initial states and expect physical ground states.

## To come

- [ ] Proper phase-map sweep: fine (f, k) grid + per-sample diagnostics
      (std, power-spectrum peak, etc.) to define morphology regions as
      measured polygons rather than hand-drawn boxes.
- [ ] Fix anisotropic mode: smaller dt, full tensor Laplacian, stability
      sweep — goal is clearly oriented labyrinths visible by eye.
- [ ] Convergence-based early stop: monitor `mean(|dB/dt|)` and halt
      per-sample when it falls below threshold.
- [ ] Raise default `n_seeds_range` to suppress bullseye patterns.
- [ ] FTH forward model: `|FFT(B * reference_hole_mask)|^2` with
      configurable reference geometry. Separate module when the real-space
      generator is nailed down.
- [ ] Dataset writer: save batches as HDF5 with metadata table, sharded
      for DataLoader consumption.

## Files

- `gs_generator.py` — main module
- `quick_check.py` — minimal demo / smoke test

## Requirements

- Python 3.10–3.12 (CuPy wheels lag for 3.13)
- `numpy`, `matplotlib`
- `cupy` (optional, GPU)