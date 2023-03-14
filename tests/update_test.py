import hydra
import jax
import pytest
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.agent.sac.sac import _update_sac_jit, create_sac_agent
from jaxman.planner.rl_planner.memory.dataset import Buffer
from jaxman.planner.rl_planner.memory.utils import _build_sample_experience
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/navigation/grid.yaml")
    )
    train_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/train/sac.yaml")
    )
    train_config.batch_size = 10
    sample_experience = _build_sample_experience(train_config)
    key = jax.random.PRNGKey(0)
    return key, env_config, sample_experience


def test_discrete_sac_update(setup):

    key, config, sample_experience = setup
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    capacity = 100

    # Discrete Env
    config.is_discrete = True
    env = JaxMANEnv(config)
    obs_space = env.observation_space
    buffer = Buffer(obs_space, env.action_space, capacity)
    buffer.size = capacity
    sac, key = create_sac_agent(obs_space, env.action_space, model_config, key)

    # update
    key, data = sample_experience(
        key, buffer, obs_space["comm"].shape[0], obs_space["comm"].shape[1], 0, 0
    )
    results = _update_sac_jit(
        key,
        sac,
        data,
        0.95,
        0.05,
        True,
        0.05,
        True,
        True,
        True,
    )


def test_continuous_sac_update(setup):

    key, config, sample_experience = setup
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    capacity = 1000

    # Continuous Env
    config.is_discrete = False
    config.map_size = 128
    env = JaxMANEnv(config)
    obs_space = env.observation_space
    buffer = Buffer(obs_space, env.action_space, capacity)
    buffer.size = capacity
    sac, key = create_sac_agent(obs_space, env.action_space, model_config, key)

    # update
    key, data = sample_experience(
        key, buffer, obs_space["comm"].shape[0], obs_space["comm"].shape[1], 0, 0
    )
    results = _update_sac_jit(
        key,
        sac,
        data,
        0.95,
        0.05,
        False,
        0.05,
        True,
        True,
        True,
    )


def test_dqn_update(setup):
    from jaxman.planner.rl_planner.agent.dqn.dqn import (
        _update_dqn_jit,
        create_dqn_agent,
    )

    key, config, sample_experience = setup
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/dqn.yaml")
    )
    capacity = 1000

    config.is_discrete = True
    env = JaxMANEnv(config)
    obs_space = env.observation_space
    buffer = Buffer(obs_space, env.action_space, capacity)
    buffer.size = capacity

    key, data = sample_experience(
        key, buffer, obs_space["comm"].shape[0], obs_space["comm"].shape[1], 0, 0
    )

    # update
    dqn, key = create_dqn_agent(obs_space, env.action_space, model_config, key)
    results = _update_dqn_jit(
        key,
        dqn,
        data,
        0.95,
        0.05,
        True,
        0.9,
        False,
        model_config.N,
        True,
        True,
        4,
    )

    # update Maxmin DQN
    results = _update_dqn_jit(
        key,
        dqn,
        data,
        0.95,
        0.05,
        True,
        0.9,
        False,
        model_config.N,
        True,
        True,
        4,
    )
