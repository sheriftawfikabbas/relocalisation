from relocalization.config import dev_config
from relocalization.sweep import beta_sweep


def _tiny():
    return dev_config(n_sites=81, t_max=20.0, dt=0.05, sample_every=10,
                      max_evals=16, n_restarts=1, ansatze=("single_sinusoid",),
                      betas=(0.0, 1.0))


def test_beta_sweep_returns_row_per_beta_ansatz():
    cfg = _tiny()
    rows = beta_sweep("A", cfg.ansatze, cfg.betas, cfg, jobs=1)
    assert len(rows) == 2
    keys = {"scenario", "beta", "ansatz", "r", "w_spread", "w_driven", "w_target",
            "relocalized", "theta_star"}
    assert keys <= set(rows[0])
    assert {r["beta"] for r in rows} == {0.0, 1.0}


def test_parallel_matches_serial():
    cfg = _tiny()
    serial = beta_sweep("A", cfg.ansatze, cfg.betas, cfg, jobs=1)
    parallel = beta_sweep("A", cfg.ansatze, cfg.betas, cfg, jobs=2)
    s = sorted(serial, key=lambda r: r["beta"])
    p = sorted(parallel, key=lambda r: r["beta"])
    for a, b in zip(s, p):
        assert a["w_driven"] == b["w_driven"]
        assert a["theta_star"] == b["theta_star"]
