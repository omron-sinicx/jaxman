import hydra
import pytest
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.memory.dataset import Buffer
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/navigation/grid.yaml")
    )
    return env_config


def test_buffer(setup):
    from jaxman.planner.rl_planner.memory.dataset import Experience
    from jaxman.planner.rl_planner.memory.utils import _build_push_experience_to_buffer

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
    push_experience_to_buffer = _build_push_experience_to_buffer(0.99, False, 4, 100)
    push_experience_to_buffer(buffer, experience)

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
    push_experience_to_buffer(buffer, experience)


def test_k_step_learning_buffer(setup):
    from jaxman.planner.rl_planner.memory.dataset import Experience
    from jaxman.planner.rl_planner.memory.utils import _build_push_experience_to_buffer

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
    push_experience_to_buffer = _build_push_experience_to_buffer(0.99, True, 4, 100)
    push_experience_to_buffer(buffer, experience)

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
    push_experience_to_buffer(buffer, experience)
