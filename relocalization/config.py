from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

GOLDEN_ALPHA = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class RunConfig:
    profile: str
    n_sites: int
    t_max: float
    dt: float
    sample_every: int
    betas: Tuple[float, ...]
    ansatze: Tuple[str, ...]
    max_evals: int
    n_restarts: int
    seed: int
    lambda_a: float
    lambda_b: float
    control_time: float
    beta_star_threshold: float
    n_segments: int
    alpha: float = GOLDEN_ALPHA
    hopping: float = 1.0
    initial_sigma_a: float = 5.0


def dev_config(**overrides) -> RunConfig:
    base = RunConfig(
        profile="dev", n_sites=201, t_max=150.0, dt=0.02, sample_every=20,
        betas=(0.0, 1.0, 3.0, 5.0), ansatze=("single_sinusoid", "two_tone"),
        max_evals=50, n_restarts=2, seed=0, lambda_a=3.0, lambda_b=1.0,
        control_time=15.0, beta_star_threshold=0.5, n_segments=8,
    )
    return replace(base, **overrides)


def production_config(**overrides) -> RunConfig:
    base = RunConfig(
        profile="production", n_sites=1201, t_max=400.0, dt=0.02, sample_every=40,
        betas=tuple([round(i * 0.1, 2) for i in range(51)] + [8.0]),
        ansatze=("single_sinusoid", "two_tone", "chirp", "bang_bang"),
        max_evals=1000, n_restarts=16, seed=0, lambda_a=3.0, lambda_b=1.0,
        control_time=15.0, beta_star_threshold=0.5, n_segments=8,
    )
    return replace(base, **overrides)
