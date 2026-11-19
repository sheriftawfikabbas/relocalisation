from __future__ import annotations

from typing import Sequence

import numpy as np

from .dnls import detect_relocalization


def restoration_score(w_spread: float, w_driven: float, w_target: float,
                      w_ref: float | None = None) -> float:
    """R = (w_spread - w_driven) / denom; measures drive-induced width reduction.

    When w_ref is None (default, scenario A): denom = w_spread - w_target, giving
    R=1 for full restoration to the target width.

    When w_ref is provided (scenario B): denom = w_ref (the beta=0 undriven spread),
    so R measures the absolute drive contribution relative to a fixed scale and is
    monotonically decreasing in beta, avoiding the re-entrance artefact that occurs
    when self-trapping shrinks w_spread at high beta.

    Returns nan when the undriven spreading is negligible (w_spread barely
    exceeds, or falls below, w_target): there is nothing to restore and the
    ratio is ill-defined.
    """
    if w_ref is not None:
        denom = w_ref
        floor = max(1e-9, 0.05 * abs(w_ref))
    else:
        denom = w_spread - w_target
        floor = max(1e-9, 0.05 * abs(w_target))
    if denom < floor:
        return float("nan")
    return float((w_spread - w_driven) / denom)


def extract_beta_star(betas: Sequence[float], r_values: Sequence[float],
                      threshold: float = 0.5) -> float:
    """Largest beta with R >= threshold, linearly interpolating the crossing.

    beta* concerns the critical *nonlinearity*, so the beta<=0 reference point
    is excluded: at beta=0 there is no nonlinearity to fight and R is degenerate
    (identically 0 in Scenario A, where w_spread == w_target). Among beta>0,
    returns nan if the smallest tested beta already fails the threshold, or
    max(beta) if all pass. Assumes R decreases with beta.
    """
    b = np.asarray(betas, dtype=float)
    r = np.asarray(r_values, dtype=float)
    order = np.argsort(b)
    b, r = b[order], r[order]
    mask = b > 0.0
    b, r = b[mask], r[mask]
    if b.size == 0 or r[0] < threshold:
        return float("nan")
    below = np.where(r < threshold)[0]
    if below.size == 0:
        return float(b[-1])
    i = int(below[0])
    b0, b1, r0, r1 = b[i - 1], b[i], r[i - 1], r[i]
    if r0 == r1:
        return float(b0)
    frac = (r0 - threshold) / (r0 - r1)
    return float(b0 + frac * (b1 - b0))


def extract_beta_star_half_peak(betas: Sequence[float], r_values: Sequence[float]) -> float:
    """beta* = largest beta where R >= half its peak value, interpolating the crossing.

    NaN entries in r_values are ignored.  Returns nan if the peak R is zero or
    only nan values are present.
    """
    b = np.asarray(betas, dtype=float)
    r = np.asarray(r_values, dtype=float)
    order = np.argsort(b)
    b, r = b[order], r[order]
    mask = np.isfinite(r)
    b, r = b[mask], r[mask]
    if b.size == 0:
        return float("nan")
    peak = float(np.max(r))
    if peak <= 0.0:
        return float("nan")
    threshold = 0.5 * peak
    return extract_beta_star(b.tolist(), r.tolist(), threshold=threshold)


def extract_beta_star_inflection(betas: Sequence[float], r_values: Sequence[float],
                                 smooth_sigma: float = 0.3) -> float:
    """beta* = inflection point of R(beta): where |dR/dbeta| is maximum.

    Gaussian-smooths R(beta) (sigma in units of beta) before differencing to
    suppress optimizer noise while preserving the physical crossover location.
    Returns nan if fewer than two finite beta>0 points are available.
    """
    from scipy.ndimage import gaussian_filter1d
    b = np.asarray(betas, dtype=float)
    r = np.asarray(r_values, dtype=float)
    order = np.argsort(b)
    b, r = b[order], r[order]
    mask = (b > 0.0) & np.isfinite(r)
    b, r = b[mask], r[mask]
    if b.size < 2:
        return float("nan")
    # Convert sigma from beta units to index units using the dense-subgrid spacing.
    # Isolated outlier points (gap > 5x median spacing) are excluded from the
    # spacing estimate so that e.g. a single beta=8 endpoint does not inflate
    # db_mean and cause the Gaussian to be applied at the wrong scale.
    diffs = np.diff(b)
    median_db = float(np.median(diffs))
    dense_mask = np.concatenate([[True], diffs <= 5.0 * median_db])
    b_dense = b[dense_mask]
    db_mean = float(np.mean(np.diff(b_dense))) if b_dense.size > 1 else float(np.mean(diffs))
    sigma_idx = smooth_sigma / db_mean if db_mean > 0 else 1.0
    r_smooth = gaussian_filter1d(r, sigma=sigma_idx)
    dr = np.diff(r_smooth)
    db = np.diff(b)
    db[db == 0] = 1e-30
    slopes = dr / db
    i = int(np.argmin(slopes))  # steepest descent
    return float(0.5 * (b[i] + b[i + 1]))


def relocalized_flag(times: np.ndarray, width: np.ndarray, t_control: float) -> bool:
    return bool(detect_relocalization(times, width, t_control)["relocalized"])
