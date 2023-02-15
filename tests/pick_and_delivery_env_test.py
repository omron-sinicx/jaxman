import hydra
import jax.numpy as jnp
from jaxman.env.navigation.env import JaxMANEnv
from omegaconf import OmegaConf


def test_grid_env():
    config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/navigation/grid.yaml")
    )

    env = JaxMANEnv(config)
    obs = env.reset()
    done = False
    base_actions = jnp.ones_like(env.sample_actions())
    for i in range(env.action_space.n):
        actions = base_actions * i
        obs, rew, done, info = env.step(actions)
        done = jnp.all(done)
