import logging
import time

import hydra
import jax
import jax.numpy as jnp
import ray
from jaxman.env.env import JaxMANEnv
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent, restore_sac_actor
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
        env = JaxMANEnv(config.env, config.seed)
        observation_space = env.observation_space
        action_space = env.action_space
        buffer = GlobalBuffer.remote(observation_space, action_space, config.train)

        key = jax.random.PRNGKey(config.seed)
        actor, critic, target_critic, temp, key = create_sac_agent(
            observation_space,
            action_space,
            config.model,
            key,
        )
        actor = restore_sac_actor(
            actor, config.env.is_discrete, config.env.is_diff_drive, "../../../model"
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
            actor,
            critic,
            target_critic,
            temp,
            action_scale,
            action_bias,
            target_entropy,
            config,
        )

        rollout_worker = RolloutWorker.remote(
            buffer,
            learner,
            actor,
            env.instance,
            config.seed,
        )
        rollout_worker.run.remote()
        evaluator = Evaluator.remote(
            learner,
            actor,
            env.instance,
            config.seed,
        )
        evaluator.run.remote()

        data_num = 0
        while data_num < config.train.batch_size:
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
        evaluator.evaluate.remote()
        done = False

        while not done:
            done = ray.get(evaluator.is_eval_done.remote())
            time.sleep(0.5)

        ray.kill(actor)
        ray.kill(evaluator)
        ray.kill(learner)
        ray.kill(buffer)
        logger.close()
        writer.close()


if __name__ == "__main__":
    main()
