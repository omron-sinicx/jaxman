import hydra
import jax.numpy as jnp
from jaxman.env.navigation.env import JaxMANEnv
from omegaconf import OmegaConf


def test_grid_env():
    config = hydra.utils.instantiate(OmegaConf.load("scripts/config/env/grid.yaml"))

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_diff_drive_env():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/diff_drive.yaml")
    )

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_continuous_env():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/continuous.yaml")
    )

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = env.sample_actions()
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = env.sample_actions()
        done = jnp.all(done)


def test_planner():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/continuous.yaml")
    )

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    actions = obs.planner_act
    while not done:
        obs, rew, done, info = env.step(actions)
        actions = obs.planner_act
        done = jnp.all(done)
