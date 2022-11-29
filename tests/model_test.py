import jax
import numpy as np
import pytest
from gym.spaces.box import Box
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent


@pytest.fixture
def setup():
    import hydra
    from omegaconf import OmegaConf

    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )

    return model_config


def test_discrete_model(setup):
    from gym.spaces.discrete import Discrete

    config = setup

    obs_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = Discrete(5)
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, config, key
    )

    obs = jax.random.normal(key, shape=(10, obs_space.shape[0]))
    action = actor.apply_fn({"params": actor.params}, obs)
    q_values = critic.apply_fn({"params": critic.params}, obs)


def test_continuous_model(setup):
    config = setup

    obs_space = Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, config, key
    )

    obs = jax.random.normal(key, shape=(10, obs_space.shape[0]))
    act_dist = actor.apply_fn({"params": actor.params}, obs)
    act = act_dist.sample(seed=key)
    q_values = critic.apply_fn({"params": critic.params}, obs, act)
