import jax.numpy as jnp
import pytest
from jaxman.env.env import JaxMANEnv


@pytest.fixture
def setup():
    import hydra
    from omegaconf import OmegaConf

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/random.yaml")
    )

    return env_config


def test_grid_env(setup):
    config = setup
    config.is_discrete = True
    config.is_diff_drive = False

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_diff_drive_env(setup):
    config = setup
    config.is_discrete = True
    config.is_diff_drive = True

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_continuous_env(setup):
    config = setup
    config.is_discrete = False
    config.is_diff_drive = False
    config.map_size = 128

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_planner(setup):
    config = setup
    config.is_discrete = False
    config.is_diff_drive = False
    config.map_size = 128

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = obs.planner_act
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = obs.planner_act
        done = jnp.all(done)
