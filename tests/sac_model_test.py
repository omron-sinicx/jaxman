import hydra
import jax
import pytest
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    return model_config


def test_discrete_model(setup):

    model_config = setup
    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/navigation/grid.yaml")
    )
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    sac, key = create_sac_agent(obs_space, act_space, model_config, key)

    obs = env.reset().split_observation()
    action = sac.actor.apply_fn({"params": sac.actor.params}, obs)
    q_values = sac.critic.apply_fn({"params": sac.critic.params}, obs)


def test_continuous_model(setup):
    model_config = setup
    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/navigation/continuous.yaml")
    )
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    sac, key = create_sac_agent(obs_space, act_space, model_config, key)

    obs = env.reset().split_observation()
    means, log_stds = sac.actor.apply_fn({"params": sac.actor.params}, obs)
