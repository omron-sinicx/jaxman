import hydra
import jax
import pytest
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from jaxman.planner.rl_planner.agent.sac.actor import create_actor
from jaxman.planner.rl_planner.rollout.pick_and_delivery.rollout import (
    build_rollout_episode,
)
from omegaconf import OmegaConf


def test_discrete_rollout():

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/grid.yaml")
    )
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/dqn.yaml")
    )
    key = jax.random.PRNGKey(0)
    env_config.timeout = 10

    # Discrete Env
    env_config.is_discrete = True
    env = JaxPandDEnv(env_config)
    actor = create_actor(env.observation_space, env.action_space, model_config, key)

    rollout_fn = build_rollout_episode(
        env.instance, actor.apply_fn, evaluate=False, model_config=model_config
    )
    carry = rollout_fn(key, actor.params, env.instance.obs)


def test_discrete_rollout():

    env_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/env/pick_and_delivery/continuous.yaml")
    )
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    key = jax.random.PRNGKey(0)
    env_config.timeout = 10

    # Continuous Env
    env_config.is_discrete = False
    env = JaxPandDEnv(env_config)
    actor = create_actor(env.observation_space, env.action_space, model_config, key)

    rollout_fn = build_rollout_episode(
        env.instance, actor.apply_fn, evaluate=False, model_config=model_config
    )
    carry = rollout_fn(key, actor.params, env.instance.obs)
