import hydra
import pytest
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.memory.dataset import Buffer
from omegaconf import OmegaConf


@pytest.fixture
def setup():

    env_config = hydra.utils.instantiate(OmegaConf.load("scripts/config/env/grid.yaml"))
    model_config = hydra.utils.instantiate(
        OmegaConf.load("scripts/config/model/sac.yaml")
    )
    return env_config, model_config


def test_rollout(setup):
    import jax
    from jaxman.planner.rl_planner.agent.sac.actor import create_actor
    from jaxman.planner.rl_planner.rollout.rollout import build_rollout_episode

    env_config, model_config = setup
    key = jax.random.PRNGKey(0)

    # Discrete Env
    env_config.is_discrete = True
    env = JaxMANEnv(env_config)
    actor = create_actor(env.observation_space, env.action_space, model_config, key)

    rollout_fn = build_rollout_episode(env.instance, actor.apply_fn, evaluate=False)
    carry = rollout_fn(key, actor.params, env.instance.obs)

    # Discrete Env
    env_config.is_discrete = False
    env_config.map_size = 128
    env = JaxMANEnv(env_config)
    actor = create_actor(env.observation_space, env.action_space, model_config, key)

    rollout_fn = build_rollout_episode(env.instance, actor.apply_fn, evaluate=False)
    carry = rollout_fn(key, actor.params, env.instance.obs)
