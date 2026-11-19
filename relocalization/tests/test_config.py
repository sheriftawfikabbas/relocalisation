from relocalization.config import dev_config, production_config, RunConfig


def test_dev_config_is_small_and_valid():
    cfg = dev_config()
    assert isinstance(cfg, RunConfig)
    assert cfg.profile == "dev"
    assert cfg.n_sites == 201
    assert cfg.betas[0] == 0.0
    assert cfg.lambda_a == 3.0 and cfg.lambda_b == 1.0
    assert cfg.n_restarts >= 1 and cfg.max_evals >= 10


def test_production_config_is_larger():
    dev, prod = dev_config(), production_config()
    assert prod.n_sites > dev.n_sites
    assert prod.t_max > dev.t_max
    assert len(prod.betas) >= len(dev.betas)
    assert len(prod.ansatze) == 4


def test_overrides_apply():
    cfg = dev_config(n_sites=51, betas=(0.0, 2.0))
    assert cfg.n_sites == 51 and cfg.betas == (0.0, 2.0)
