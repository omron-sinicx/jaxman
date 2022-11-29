import logging
import time

import hydra
import jax
import jax.numpy as jnp
import ray
from flax.training import checkpoints
from jaxman.env.env import JaxMANEnv
from jaxman.env.viz import render_gif
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent
from jaxman.planner.rl_planner.logger import LogResult
from jaxman.planner.rl_planner.rollout.rollout import build_rollout_episode


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


@hydra.main(config_path="config", config_name="train_rl")
def main(config):
    ray.init()
    logger = logging.getLogger("root")
    logger.addFilter(CheckTypesFilter())

    # initialize enrironment
    env = JaxMANEnv(config.env, config.seed)
    observation_space = env.observation_space
    action_space = env.action_space

    key = jax.random.PRNGKey(config.seed)
    actor, _, _, _, key = create_sac_agent(
        observation_space,
        action_space,
        config.model,
        key,
    )
    if config.env.is_discrete:
        if config.env.is_diff_drive:
            actor = checkpoints.restore_checkpoint(
                ckpt_dir="../../../model", target=actor, prefix="diff_drive_actor"
            )
        else:
            actor = checkpoints.restore_checkpoint(
                ckpt_dir="../../../model", target=actor, prefix="grid_actor"
            )
    else:
        actor = checkpoints.restore_checkpoint(
            ckpt_dir="../../../model", target=actor, prefix="continuous_actor"
        )

    rollout_fn = build_rollout_episode(env.instance, actor.apply_fn, evaluate=True)
    # rollout episode
    carry = rollout_fn(key, actor.params, env.instance.obs)
    steps = carry.episode_steps
    state_traj = carry.experience.observations[:steps, :, :5]
    last_state = carry.state.cat()
    state_traj = jnp.vstack((state_traj, jnp.expand_dims(last_state, axis=0)))
    dones = carry.experience.dones[:steps, :]
    last_dones = jnp.ones_like(dones[0], dtype=bool)
    dones = jnp.vstack((dones, jnp.expand_dims(last_dones, 0)))
    env.reset(task_info=carry.task_info)
    env.save_gif(state_traj, dones)


if __name__ == "__main__":
    main()
