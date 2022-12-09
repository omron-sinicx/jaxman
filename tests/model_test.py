import jax
import numpy as np
import pytest
from gym.spaces.box import Box
from jaxman.env.env import JaxMANEnv
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent


@pytest.fixture
def setup():
    import hydra
    from omegaconf import OmegaConf

    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/random.yaml")
    )

    return model_config, env_config


def test_discrete_model(setup):

    model_config, env_config = setup

    env_config.is_discrete = True
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, model_config, key
    )

    obs = jax.random.normal(key, shape=(10, obs_space["obs"].shape[0]))
    comm = jax.random.normal(key, shape=(10, *obs_space["comm"].shape))
    action = actor.apply_fn({"params": actor.params}, obs, comm)
    q_values = critic.apply_fn({"params": critic.params}, obs, comm)


def test_continuous_model(setup):
    model_config, env_config = setup

    env_config.is_discrete = False
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, model_config, key
    )

    obs = jax.random.normal(key, shape=(10, obs_space["obs"].shape[0]))
    comm = jax.random.normal(key, shape=(10, *obs_space["comm"].shape))
    act_dist = actor.apply_fn({"params": actor.params}, obs, comm)
    act = act_dist.sample(seed=key)
    q_values = critic.apply_fn({"params": critic.params}, obs, comm, act)
