"""
Gray-Scott reaction-diffusion generator for magnetic-domain-like patterns.

Produces continuous-valued 2D fields morphologically similar to out-of-plane
magnetized thin-film domain configurations (labyrinths, bubbles, stripes,
mixed phases) for use as synthetic ground truth in ML pipelines (FTH/CDI
quality assessment, inpainting, etc.).

Design notes
------------
- Batched on GPU: a single step costs almost the same for batch=1 vs batch=64,
  so we always carry a batch dimension internally.
- CuPy preferred, NumPy fallback for debugging / single-sample reference.
- Anisotropic diffusion via a rotated/stretched Laplacian stencil (per-sample
  theta, alpha).
- Spatially-varying (f, k) via low-frequency Gaussian random fields generated
  in Fourier space.
- Randomized seeding, parameter draws from the "interesting crescent" of
  Gray-Scott, and optional early stopping anywhere on the relaxation path so
  transient/metastable states are first-class outputs, not just fully-relaxed
  labyrinths.

The underlying PDE (Gray-Scott):

    dA/dt = D_A * laplacian(A) - A*B^2 + f*(1 - A)
    dB/dt = D_B * laplacian(B) + A*B^2 - (k + f)*B

Discretized with explicit Euler, dt = 1.0, Laplacian stencil
    [[0.05, 0.2, 0.05],
     [0.20,-1.0, 0.20],
     [0.05, 0.2, 0.05]]
as used by Karl Sims' RD Tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as _np

# ---------------------------------------------------------------------------
# Array-library handling: prefer CuPy, fall back to NumPy transparently.
# ---------------------------------------------------------------------------
try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    _cp = None
    _HAS_CUPY = False


def _xp(use_gpu: bool):
    """Return the active array module (cupy or numpy)."""
    if use_gpu and _HAS_CUPY:
        return _cp
    return _np


def _to_numpy(arr):
    """Move a cupy array to numpy; no-op for numpy arrays."""
    if _HAS_CUPY and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return _np.asarray(arr)


# ---------------------------------------------------------------------------
# Parameter sampling: the interesting crescent in (k, f) space.
# ---------------------------------------------------------------------------
#
# Karl Sims' pattern map spans roughly f in [0.01, 0.10], k in [0.045, 0.070].
# Named "morphology classes" below are approximate regions. They overlap and
# boundaries are soft; the labels are for convenience when building datasets.

MORPHOLOGY_REGIONS = {
    # (f_min, f_max, k_min, k_max)
    # Tightened after empirical testing to stay well inside the non-trivial
    # region of the Gray-Scott phase diagram. Previously "labyrinth" was
    # too loose and ~20% of samples collapsed to uniform states.
    #
    # CHECK THIS AGAIN!!!
    #
    "labyrinth":    (0.036, 0.050, 0.058, 0.063),  # classic stripes/maze
    "bubbles":      (0.020, 0.035, 0.053, 0.060),  # isolated spots of B
    "mitosis":      (0.030, 0.042, 0.060, 0.065),  # dividing spots (dynamic)
    "coral":        (0.048, 0.062, 0.059, 0.063),  # branching growth
    "stripes":      (0.025, 0.042, 0.055, 0.060),  # elongated domains
    "solitons":     (0.022, 0.032, 0.056, 0.061),  # traveling spots
    "holes":        (0.038, 0.055, 0.057, 0.061),  # labyrinth-in-reverse
}


def sample_fk(
    n: int,
    regions: Optional[Union[str, list]] = None,
    rng: Optional[_np.random.Generator] = None,
) -> _np.ndarray:
    """Draw n (f, k) pairs, optionally restricted to named morphology regions.

    Parameters
    ----------
    n : int
        Number of samples.
    regions : str | list[str] | None
        Morphology class name(s). If None, samples broadly across the
        interesting crescent.
    rng : np.random.Generator | None
        Random number generator (CPU-side).

    Returns
    -------
    fk : np.ndarray of shape (n, 2), columns = (f, k).
    """
    rng = rng if rng is not None else _np.random.default_rng()

    if regions is None:
        # Broad sampling: f in [0.014, 0.07], k in [0.045, 0.068],
        # then reject points clearly outside the interesting crescent.
        f = rng.uniform(0.014, 0.07, size=n * 4)
        k = rng.uniform(0.045, 0.068, size=n * 4)
        # Crude crescent mask: approximate upper and lower bounds.
        # f_upper(k) ~ 0.09 - 5*(k-0.045), f_lower(k) ~ 0.005 + 3*(k-0.045)
        upper = 0.09 - 5.0 * (k - 0.045)
        lower = 0.005 + 3.0 * (k - 0.045)
        mask = (f < upper) & (f > lower)
        fk = _np.stack([f[mask], k[mask]], axis=1)
        if fk.shape[0] >= n:
            return fk[:n]
        # Fallback: top up with unmasked draws
        extra = rng.uniform(
            low=[0.020, 0.050], high=[0.060, 0.065], size=(n - fk.shape[0], 2)
        )
        return _np.vstack([fk, extra])

    if isinstance(regions, str):
        regions = [regions]

    out = _np.empty((n, 2), dtype=_np.float64)
    # Assign each sample to a region uniformly at random
    region_idx = rng.integers(0, len(regions), size=n)
    for i, ridx in enumerate(region_idx):
        name = regions[ridx]
        if name not in MORPHOLOGY_REGIONS:
            raise ValueError(
                f"Unknown region '{name}'. Options: {list(MORPHOLOGY_REGIONS)}"
            )
        fmin, fmax, kmin, kmax = MORPHOLOGY_REGIONS[name]
        out[i, 0] = rng.uniform(fmin, fmax)
        out[i, 1] = rng.uniform(kmin, kmax)
    return out


# ---------------------------------------------------------------------------
# Low-frequency Gaussian random fields for spatially-varying (f, k).
# ---------------------------------------------------------------------------

def _grf_2d(batch, H, W, correlation_length, xp, rng_seed=None):
    """Generate batched 2D Gaussian random fields with a given correlation
    length (in pixels), normalized to zero mean and unit std per sample.

    Uses Fourier-domain filtering with a Gaussian spectral envelope, which
    matches Perlin-like smooth noise well enough for our purposes and is
    cheap on GPU.
    """
    if rng_seed is not None:
        if xp is _np:
            rng = _np.random.default_rng(rng_seed)
            white = rng.standard_normal((batch, H, W)).astype(_np.float32)
        else:
            # CuPy: use its own RNG with the provided seed
            state = _cp.random.RandomState(rng_seed)
            white = state.standard_normal((batch, H, W), dtype=_cp.float32)
    else:
        white = xp.random.standard_normal((batch, H, W)).astype(xp.float32)

    # Build Gaussian spectral filter (radial)
    fy = xp.fft.fftfreq(H).astype(xp.float32)[:, None]
    fx = xp.fft.fftfreq(W).astype(xp.float32)[None, :]
    r2 = fx * fx + fy * fy
    # sigma_freq sets correlation length: larger sigma_freq -> shorter corr.
    sigma_freq = 1.0 / max(correlation_length, 1.0)
    env = xp.exp(-0.5 * r2 / (sigma_freq * sigma_freq))

    spec = xp.fft.fft2(white) * env[None, :, :]
    field_ = xp.fft.ifft2(spec).real

    # Per-sample normalization
    m = field_.mean(axis=(1, 2), keepdims=True)
    s = field_.std(axis=(1, 2), keepdims=True) + 1e-8
    return (field_ - m) / s


# ---------------------------------------------------------------------------
# Laplacian: isotropic and anisotropic variants.
# ---------------------------------------------------------------------------

def _laplacian_iso(u, xp):
    """3x3 Karl-Sims stencil Laplacian with periodic BC.

    Implements:
        L[i,j] = 0.2  * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
              + 0.05 * (u[i-1,j-1] + u[i-1,j+1] + u[i+1,j-1] + u[i+1,j+1])
              - 1.0  * u[i,j]

    Vectorized via xp.roll (works for any batch shape (..., H, W)).
    """
    up    = xp.roll(u, -1, axis=-2)
    down  = xp.roll(u,  1, axis=-2)
    left  = xp.roll(u,  1, axis=-1)
    right = xp.roll(u, -1, axis=-1)
    ul    = xp.roll(up,    1, axis=-1)
    ur    = xp.roll(up,   -1, axis=-1)
    dl    = xp.roll(down,  1, axis=-1)
    dr    = xp.roll(down, -1, axis=-1)
    return 0.2 * (up + down + left + right) \
         + 0.05 * (ul + ur + dl + dr) \
         - u


def _laplacian_aniso(u, theta, alpha, xp):
    """Anisotropic Laplacian: diffusion faster along direction theta than
    perpendicular, with anisotropy ratio alpha = D_perp / D_parallel in (0, 1].

    Implementation: blend the isotropic Sims-style Laplacian L_iso with a
    rotation-aligned anisotropic correction. This preserves the magnitude
    calibration (and hence dt=1, D=1 stability) of the isotropic case while
    adding a directional bias.

        L_aniso = L_iso  +  (1 - alpha) * [ (cos(2 theta) (L_xx - L_yy)/2
                                             + sin(2 theta)  L_xy ) ]
                             * scale

    where L_xx, L_yy, L_xy are normalized second-derivative operators with
    the same characteristic magnitude as the Sims stencil. This adds only
    a direction-dependent correction; when alpha = 1 it reduces to L_iso.
    """
    # Base isotropic Laplacian (Sims stencil, same magnitude calibration).
    L_iso = _laplacian_iso(u, xp)

    # Directional second derivatives, scaled to roughly match Sims-stencil
    # magnitude. A raw (uxx - uyy) has magnitude ~2x the Sims Laplacian for
    # sharp gradients; we apply a 0.25 prefactor so the correction is small
    # relative to the base Laplacian and remains stable at dt=1.
    uxx = xp.roll(u, -1, axis=-1) - 2.0 * u + xp.roll(u, 1, axis=-1)
    uyy = xp.roll(u, -1, axis=-2) - 2.0 * u + xp.roll(u, 1, axis=-2)
    uxy = 0.25 * (
        xp.roll(xp.roll(u, -1, axis=-2), -1, axis=-1)
        - xp.roll(xp.roll(u, -1, axis=-2),  1, axis=-1)
        - xp.roll(xp.roll(u,  1, axis=-2), -1, axis=-1)
        + xp.roll(xp.roll(u,  1, axis=-2),  1, axis=-1)
    )

    c2 = xp.cos(2.0 * theta)[:, None, None]
    s2 = xp.sin(2.0 * theta)[:, None, None]
    a  = alpha[:, None, None]

    # (1 - alpha) sets anisotropy strength; 0.25 scales the correction into
    # the same magnitude band as L_iso to remain stable with dt = 1.
    correction = 0.25 * (1.0 - a) * (c2 * 0.5 * (uxx - uyy) + s2 * uxy)
    return L_iso + correction


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _make_seeds(batch, H, W, n_seeds_range, radius_range, xp, rng_seed=None):
    """Create initial B field: zeros with a few round patches set to ~1.

    Returns shape (batch, H, W), float32.
    """
    B0 = xp.zeros((batch, H, W), dtype=xp.float32)
    # We use CPU numpy RNG for indices (cheap, avoids per-sample CuPy RNG
    # overhead) and assign to xp array.
    rng = _np.random.default_rng(rng_seed)
    for b in range(batch):
        n_seeds = int(rng.integers(n_seeds_range[0], n_seeds_range[1] + 1))
        for _ in range(n_seeds):
            cy = int(rng.integers(0, H))
            cx = int(rng.integers(0, W))
            r = int(rng.integers(radius_range[0], radius_range[1] + 1))
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            # Circular mask inside the bounding box
            yy, xx = _np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
            mask = (yy * yy + xx * xx) <= (r * r)
            if xp is _np:
                B0[b, y0:y1, x0:x1][mask] = 1.0
            else:
                # CuPy doesn't support boolean-indexed assignment from a numpy
                # mask the same way; assign via a cupy mask.
                cmask = _cp.asarray(mask)
                patch = B0[b, y0:y1, x0:x1]
                patch[cmask] = 1.0
                B0[b, y0:y1, x0:x1] = patch
    return B0


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

@dataclass
class GrayScottConfig:
    """Configuration for a batched Gray-Scott generation run."""

    # Grid
    H: int = 256
    W: int = 256
    batch: int = 16

    # Core Gray-Scott parameters
    D_A: float = 1.0
    D_B: float = 0.5
    dt: float = 1.0

    # Feed/kill: either scalar-per-sample draws, or spatially-varying fields.
    # If sample_region is given, (f, k) means are drawn from that region.
    sample_region: Optional[Union[str, list]] = None  # e.g. "labyrinth"
    f_override: Optional[_np.ndarray] = None  # shape (batch,) if provided
    k_override: Optional[_np.ndarray] = None  # shape (batch,) if provided

    # Spatially-varying (f, k): enable per-field; amplitude is fraction of mean
    fk_spatial: bool = False
    fk_correlation_length: float = 48.0  # pixels
    f_spatial_amp: float = 0.15  # +/- fraction of f_mean
    k_spatial_amp: float = 0.05  # smaller; k is more sensitive

    # Anisotropic diffusion
    anisotropic: bool = False
    alpha_range: Tuple[float, float] = (0.3, 1.0)  # 1.0 = isotropic
    theta_range: Tuple[float, float] = (0.0, _np.pi)  # radians

    # Seeding
    n_seeds_range: Tuple[int, int] = (3, 12)
    seed_radius_range: Tuple[int, int] = (3, 8)

    # Time evolution
    n_steps: int = 8000  # typical for labyrinth convergence at 256x256
    record_every: int = 0  # if > 0, also record intermediate frames
    # For "diverse transient states" sampling:
    random_stop: bool = False  # if True, each sample stops at a random step
    random_stop_min_frac: float = 0.3  # earliest stop as fraction of n_steps

    # GPU
    use_gpu: bool = True

    # RNG
    seed: Optional[int] = None


class GrayScottBatch:
    """Batched Gray-Scott simulator producing continuous (A, B) fields."""

    def __init__(self, cfg: GrayScottConfig):
        self.cfg = cfg
        self.xp = _xp(cfg.use_gpu)
        self._rng = _np.random.default_rng(cfg.seed)

        # Resolve per-sample (f, k) means
        if cfg.f_override is not None and cfg.k_override is not None:
            f_mean = _np.asarray(cfg.f_override, dtype=_np.float32)
            k_mean = _np.asarray(cfg.k_override, dtype=_np.float32)
            assert f_mean.shape == (cfg.batch,)
            assert k_mean.shape == (cfg.batch,)
        else:
            fk = sample_fk(cfg.batch, regions=cfg.sample_region, rng=self._rng)
            f_mean = fk[:, 0].astype(_np.float32)
            k_mean = fk[:, 1].astype(_np.float32)

        self.f_mean_cpu = f_mean.copy()
        self.k_mean_cpu = k_mean.copy()

        xp = self.xp
        self.f_mean = xp.asarray(f_mean)
        self.k_mean = xp.asarray(k_mean)

        # Build (f, k) fields
        if cfg.fk_spatial:
            grf_f = _grf_2d(
                cfg.batch, cfg.H, cfg.W,
                cfg.fk_correlation_length, xp,
                rng_seed=(cfg.seed + 1) if cfg.seed is not None else None,
            )
            grf_k = _grf_2d(
                cfg.batch, cfg.H, cfg.W,
                cfg.fk_correlation_length, xp,
                rng_seed=(cfg.seed + 2) if cfg.seed is not None else None,
            )
            self.f_field = self.f_mean[:, None, None] * (
                1.0 + cfg.f_spatial_amp * grf_f
            )
            self.k_field = self.k_mean[:, None, None] * (
                1.0 + cfg.k_spatial_amp * grf_k
            )
        else:
            self.f_field = self.f_mean[:, None, None]  # broadcasts
            self.k_field = self.k_mean[:, None, None]

        # Anisotropy
        if cfg.anisotropic:
            theta = self._rng.uniform(
                cfg.theta_range[0], cfg.theta_range[1], size=cfg.batch
            ).astype(_np.float32)
            alpha = self._rng.uniform(
                cfg.alpha_range[0], cfg.alpha_range[1], size=cfg.batch
            ).astype(_np.float32)
            self.theta = xp.asarray(theta)
            self.alpha = xp.asarray(alpha)
        else:
            self.theta = None
            self.alpha = None

        # Initial state
        self.A = xp.ones((cfg.batch, cfg.H, cfg.W), dtype=xp.float32)
        self.B = _make_seeds(
            cfg.batch, cfg.H, cfg.W,
            cfg.n_seeds_range, cfg.seed_radius_range, xp,
            rng_seed=cfg.seed,
        )

        # If random_stop, draw per-sample stop steps
        if cfg.random_stop:
            lo = int(cfg.random_stop_min_frac * cfg.n_steps)
            self.stop_steps = self._rng.integers(
                lo, cfg.n_steps + 1, size=cfg.batch
            )
            # A frozen mask: 1 while still evolving, 0 once stopped.
            self._active = xp.ones(cfg.batch, dtype=xp.float32)
            self._A_frozen = self.A.copy()
            self._B_frozen = self.B.copy()
        else:
            self.stop_steps = None

        self.frames = [] if cfg.record_every > 0 else None
        self._step_counter = 0

    # -- core step -----------------------------------------------------------
    def _laplacian(self, u):
        if self.cfg.anisotropic:
            return _laplacian_aniso(u, self.theta, self.alpha, self.xp)
        return _laplacian_iso(u, self.xp)

    def step(self, n: int = 1):
        xp = self.xp
        A, B = self.A, self.B
        D_A, D_B, dt = self.cfg.D_A, self.cfg.D_B, self.cfg.dt
        f_field, k_field = self.f_field, self.k_field

        for _ in range(n):
            LA = self._laplacian(A)
            LB = self._laplacian(B)
            ABB = A * B * B
            dA = D_A * LA - ABB + f_field * (1.0 - A)
            dB = D_B * LB + ABB - (k_field + f_field) * B
            A = A + dt * dA
            B = B + dt * dB
            # Numerical safety: clamp to physical range
            A = xp.clip(A, 0.0, 1.0)
            B = xp.clip(B, 0.0, 1.0)

            self._step_counter += 1

            # Handle random_stop: freeze samples whose stop step has passed
            if self.stop_steps is not None:
                just_stopped = (self.stop_steps == self._step_counter)
                if _np.any(just_stopped):
                    idx = _np.where(just_stopped)[0]
                    # Store their current fields into frozen buffers
                    for b in idx:
                        self._A_frozen[int(b)] = A[int(b)]
                        self._B_frozen[int(b)] = B[int(b)]
                        self._active[int(b)] = 0.0

            if self.frames is not None and (
                self._step_counter % self.cfg.record_every == 0
            ):
                self.frames.append(_to_numpy(B.copy()))

        self.A, self.B = A, B

    def run(self) -> Tuple[_np.ndarray, _np.ndarray]:
        """Run the full schedule and return (A, B) as numpy arrays."""
        self.step(self.cfg.n_steps)
        if self.stop_steps is not None:
            # Use frozen states for stopped samples, current for the rest
            active = self._active.astype(bool) if _HAS_CUPY else self._active.astype(bool)
            # Easier: just overwrite with frozen where stop_steps <= counter
            A_out = self.xp.where(
                self._active[:, None, None] > 0.5, self.A, self._A_frozen
            )
            B_out = self.xp.where(
                self._active[:, None, None] > 0.5, self.B, self._B_frozen
            )
            return _to_numpy(A_out), _to_numpy(B_out)
        return _to_numpy(self.A), _to_numpy(self.B)


# ---------------------------------------------------------------------------
# High-level convenience functions
# ---------------------------------------------------------------------------

def generate(
    batch: int = 16,
    H: int = 256,
    W: int = 256,
    n_steps: int = 8000,
    region: Optional[Union[str, list]] = None,
    anisotropic: bool = False,
    fk_spatial: bool = False,
    random_stop: bool = False,
    seed: Optional[int] = None,
    use_gpu: bool = True,
    **overrides,
) -> Tuple[_np.ndarray, _np.ndarray, dict]:
    """One-call generator. Returns (A, B, meta).

    `meta` is a dict of per-sample parameters (f, k, theta, alpha, stop_step)
    so you can log/condition on them downstream.
    """
    cfg = GrayScottConfig(
        H=H, W=W, batch=batch, n_steps=n_steps,
        sample_region=region,
        anisotropic=anisotropic,
        fk_spatial=fk_spatial,
        random_stop=random_stop,
        seed=seed,
        use_gpu=use_gpu,
        **overrides,
    )
    sim = GrayScottBatch(cfg)
    A, B = sim.run()
    meta = {
        "f": sim.f_mean_cpu,
        "k": sim.k_mean_cpu,
        "theta": _to_numpy(sim.theta) if sim.theta is not None else None,
        "alpha": _to_numpy(sim.alpha) if sim.alpha is not None else None,
        "stop_steps": sim.stop_steps,
        "n_steps": n_steps,
        "H": H,
        "W": W,
    }
    return A, B, meta


# ---------------------------------------------------------------------------
# Self-test: run a tiny NumPy simulation and check it produces structure.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Force NumPy so this runs anywhere
    A, B, meta = generate(
        batch=4, H=128, W=128, n_steps=3000,
        region="labyrinth",
        use_gpu=False, seed=0,
    )
    print(f"A shape: {A.shape}, dtype: {A.dtype}")
    print(f"B shape: {B.shape}, dtype: {B.dtype}")
    print(f"f values: {meta['f']}")
    print(f"k values: {meta['k']}")
    print(f"B stats per sample:")
    for i in range(A.shape[0]):
        print(
            f"  sample {i}: B mean={B[i].mean():.4f} "
            f"std={B[i].std():.4f} min={B[i].min():.4f} max={B[i].max():.4f}"
        )
