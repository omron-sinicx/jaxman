"""script for training models

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
import logging
import time

import hydra
import jax
import jax.numpy as jnp
import ray
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from jaxman.planner.rl_planner.agent.core import create_agent, restore_agent
from jaxman.planner.rl_planner.logger import LogResult
from jaxman.planner.rl_planner.worker import (
    Evaluator,
    GlobalBuffer,
    Learner,
    RolloutWorker,
)
from tensorboardX import SummaryWriter


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


@hydra.main(config_path="config", config_name="train_rl")
def main(config):
    ray.init()
    logger = logging.getLogger("root")
    logger.addFilter(CheckTypesFilter())

    for seed in range(config.seed_start, config.seed_end):
        config.seed = seed

        writer = SummaryWriter(logdir=f"./tb/seed_{seed}")

        logger = LogResult(writer, config)
        # initialize enrironment
        if config.env.env_name == "navigation":
            env = JaxMANEnv(config.env, config.seed)
        else:
            env = JaxPandDEnv(config.env, config.seed)
        observation_space = env.observation_space
        action_space = env.action_space
        buffer = GlobalBuffer.remote(observation_space, action_space, config)

        key = jax.random.PRNGKey(config.seed)
        agent, key = create_agent(
            observation_space,
            action_space,
            config.model,
            key,
        )
        if config.train.use_pretrained_model:
            agent = restore_agent(
                agent,
                config.env.is_discrete,
                config.env.is_diff_drive,
                config.model,
                f"../../../../../model/{config.env.env_name}_{config.env.obs_type}",
            )

        action_scale = None
        action_bias = None
        if config.env.is_discrete:
            target_entropy = (
                -jnp.log(1.0 / env.action_space.n) * config.train.target_entropy_ratio
            )
        else:
            target_entropy = -env.action_space.shape[0] / 2
        # target_entropy = -action_space.n/2
        learner = Learner.remote(
            buffer,
            agent,
            action_scale,
            action_bias,
            target_entropy,
            config,
        )

        rollout_worker = RolloutWorker.remote(
            buffer,
            learner,
            agent.actor,
            env.instance,
            config.model,
            config.seed,
        )
        rollout_worker.run.remote()
        evaluator = Evaluator.remote(
            learner,
            agent.actor,
            env.instance,
            config.model,
            config.seed,
        )
        evaluator.run.remote()

        data_num = 0
        while data_num < config.train.initial_rollout:
            time.sleep(1)
            data_num = ray.get(ray.get(buffer.num_data.remote()))

        buffer.run.remote()
        learner.run.remote()

        log_interval = 10

        done = False
        while not done:
            time.sleep(log_interval)
            done, loss_info = ray.get(learner.stats.remote(log_interval))
            rollout_reward = ray.get(rollout_worker.stats.remote(log_interval))
            (
                eval_reward,
                trial_info,
                animation,
            ) = ray.get(evaluator.stats.remote(log_interval))
            logger.log_result(
                rollout_reward,
                eval_reward,
                trial_info,
                loss_info,
                animation,
            )
            print()
        evaluator.evaluate.remote(config.train.eval_iters)
        done = False

        # while not done:
        #     done = ray.get(evaluator.is_eval_done.remote())
        #     time.sleep(0.5)

        ray.kill(rollout_worker)
        ray.kill(evaluator)
        ray.kill(learner)
        ray.kill(buffer)
        writer.close()


if __name__ == "__main__":
    main()
