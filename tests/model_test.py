import hydra
import jax
import pytest
from jaxman.env.env import JaxMANEnv
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
    env_config = hydra.utils.instantiate(OmegaConf.load("scripts/config/env/grid.yaml"))
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, model_config, key
    )

    obs = jax.random.normal(key, shape=(3, *obs_space["obs"].shape))
    comm = jax.random.normal(key, shape=(3, *obs_space["comm"].shape))
    mask = jax.random.normal(key, shape=(3, *obs_space["mask"].shape))
    action = actor.apply_fn({"params": actor.params}, obs, comm, mask)
    q_values = critic.apply_fn({"params": critic.params}, obs, comm, mask)


def test_continuous_model(setup):
    model_config = setup
    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/continuous.yaml")
    )
    env = JaxMANEnv(env_config)

    obs_space = env.observation_space
    act_space = env.action_space
    key = jax.random.PRNGKey(0)

    actor, critic, target_critic, temp, key = create_sac_agent(
        obs_space, act_space, model_config, key
    )

    obs = jax.random.normal(key, shape=(10, *obs_space["obs"].shape))
    comm = jax.random.normal(key, shape=(10, *obs_space["comm"].shape))
    mask = jax.random.normal(key, shape=(10, *obs_space["mask"].shape))
    act_dist = actor.apply_fn({"params": actor.params}, obs, comm, mask)
    act = act_dist.sample(seed=key)
    q_values = critic.apply_fn({"params": critic.params}, obs, comm, mask, act)
