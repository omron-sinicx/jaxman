import hydra
import pytest
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.memory.dataset import Buffer
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    env_config = hydra.utils.instantiate(OmegaConf.load("scripts/config/env/grid.yaml"))
    return env_config


def test_buffer(setup):
    from jaxman.planner.rl_planner.memory.dataset import Experience
    from jaxman.planner.rl_planner.memory.utils import _push_experience_to_buffer

    config = setup

    # Discrete Env
    config.is_discrete = True
    env = JaxMANEnv(config)
    buffer = Buffer(env.observation_space, env.action_space)
    observations = env.reset().cat()
    actions = env.sample_actions()

    experience = Experience.reset(10, 10, observations, actions)
    dones = experience.dones
    # add done one last
    dones = dones.at[-2].set(1 - dones[-1, :])
    experience = experience._replace(dones=dones)
    _push_experience_to_buffer(buffer, experience)

    # Continuous Env
    config.is_discrete = False
    config.map_size = 128
    env = JaxMANEnv(config)
    buffer = Buffer(env.observation_space, env.action_space)
    observations = env.reset().cat()
    actions = env.sample_actions()

    experience = Experience.reset(10, 10, observations, actions)
    dones = experience.dones
    # add done one last
    dones = dones.at[-2].set(1 - dones[-1, :])
    experience = experience._replace(dones=dones)
    _push_experience_to_buffer(buffer, experience)


def test_update(setup):
    import jax
    from jaxman.planner.rl_planner.agent.sac.sac import (
        _update_sac_jit,
        create_sac_agent,
    )
    from jaxman.planner.rl_planner.memory.utils import _sample_experience

    config = setup
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    key = jax.random.PRNGKey(0)
    capacity = 1000

    # Discrete Env
    config.is_discrete = True
    env = JaxMANEnv(config)
    obs_space = env.observation_space
    buffer = Buffer(obs_space, env.action_space, capacity)
    buffer.size = capacity
    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, env.action_space, model_config, key
    )

    # update
    data = _sample_experience(
        buffer, 10, obs_space["comm"].shape[0], obs_space["comm"].shape[1]
    )
    results = _update_sac_jit(
        key,
        actor,
        critic,
        target_critic,
        temp,
        data,
        0.95,
        0.05,
        True,
        0.05,
        True,
        True,
        True,
    )

    # Continuous Env
    config.is_discrete = False
    config.map_size = 128
    env = JaxMANEnv(config)
    obs_space = env.observation_space
    buffer = Buffer(obs_space, env.action_space, capacity)
    buffer.size = capacity
    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, env.action_space, model_config, key
    )

    # update
    data = _sample_experience(
        buffer, 10, obs_space["comm"].shape[0], obs_space["comm"].shape[1]
    )
    results = _update_sac_jit(
        key,
        actor,
        critic,
        target_critic,
        temp,
        data,
        0.95,
        0.05,
        False,
        0.05,
        True,
        True,
        True,
    )
