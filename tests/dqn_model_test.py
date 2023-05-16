import hydra
import jax
import pytest
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from jaxman.planner.rl_planner.agent.dqn.dqn import create_dqn_agent
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/dqn.yaml")
    )
    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/grid.yaml")
    )
    env = JaxPandDEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    obs = env.reset().split_observation()
    key = jax.random.PRNGKey(0)
    return obs, obs_space, act_space, model_config, key


def test_model(setup):

    obs, obs_space, act_space, model_config, key = setup

    dqn, key = create_dqn_agent(obs_space, act_space, model_config, key)
    q_values = dqn.actor.apply_fn({"params": dqn.actor.params}, obs)
