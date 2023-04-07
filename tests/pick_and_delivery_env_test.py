import hydra
import jax.numpy as jnp
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from omegaconf import OmegaConf


def test_grid_env():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/grid.yaml")
    )

    env = JaxPandDEnv(config)
    obs = env.reset()
    done = False
    base_actions = jnp.ones_like(env.sample_actions())
    for i in range(env.action_space.n):
        actions = base_actions * i
        obs, rew, done, info = env.step(actions)
        done = jnp.all(done)


def test_continuous_env():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/continuous.yaml")
    )

    env = JaxPandDEnv(config)
    obs = env.reset()
    done = False
    for i in range(3):
        actions = env.sample_actions()
        obs, rew, done, info = env.step(actions)
        done = jnp.all(done)
