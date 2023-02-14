import hydra
import jax
import pytest
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from jaxman.planner.rl_planner.agent.sac.actor import create_actor
from jaxman.planner.rl_planner.rollout.pick_and_delivery.rollout import (
    build_rollout_episode,
)
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/grid.yaml")
    )
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/dqn.yaml")
    )
    key = jax.random.PRNGKey(0)
    return key, env_config, model_config


def test_discrete_rollout(setup):

    key, env_config, model_config = setup

    # Discrete Env
    env_config.is_discrete = True
    env = JaxPandDEnv(env_config)
    actor = create_actor(env.observation_space, env.action_space, model_config, key)

    rollout_fn = build_rollout_episode(
        env.instance, actor.apply_fn, evaluate=False, model_name=model_config.name
    )
    carry = rollout_fn(key, actor.params, env.instance.obs)
