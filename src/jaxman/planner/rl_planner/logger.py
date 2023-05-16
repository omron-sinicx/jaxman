"""Definition of logger, log training and evaluation data

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import jax.numpy as jnp
import numpy as np


class LogResult:
    def __init__(self, writer, config) -> None:
        self.total_episodes = 0
        self.total_updates = 0
        self.writer = writer
        self.env_name = config.env.env_name

    def log_result(
        self,
        reward,
        eval_reward,
        trial_info,
        loss_info,
        animation,
    ):
        # log loss informatino
        if loss_info:
            for k, v in loss_info[0].items():
                self.writer.add_scalar(f"loss/{k}", jnp.mean(v), self.total_updates)
            self.total_updates += len(loss_info)

        # log training data reward
        if reward is not None:
            self.writer.add_scalar(
                f"train/reward", np.mean(reward), self.total_episodes
            )

        # log evaluation data
        if trial_info:
            ##### reward #####
            # mean
            self.writer.add_scalar(
                f"evaluation/reward", np.mean(eval_reward), self.total_episodes
            )
            # TODO
            for i in range(len(trial_info)):
                self.total_episodes += 1
                ##### trial information #####
                collided = sum(trial_info[i].agent_collided)
                self.writer.add_scalar(
                    "evaluation/collided", collided, self.total_episodes
                )
                solved = sum(trial_info[i].solved)
                self.writer.add_scalar("evaluation/solved", solved, self.total_episodes)
                if is_success and self.env_name == "navigation":
                    timeout = sum(trial_info[i].timeout)
                    self.writer.add_scalar(
                        "evaluation/timeout", timeout, self.total_episodes
                    )
                is_success = trial_info[i].is_success
                self.writer.add_scalar(
                    "evaluation/is_success",
                    is_success.astype(float),
                    self.total_episodes,
                )
                if is_success and self.env_name == "navigation":
                    makespan = trial_info[i].makespan
                    self.writer.add_scalar(
                        "evaluation/makespan",
                        makespan,
                        self.total_episodes,
                    )
                    sum_of_cost = trial_info[i].sum_of_cost
                    self.writer.add_scalar(
                        "evaluation/sum_of_cost",
                        sum_of_cost,
                        self.total_episodes,
                    )

        if animation is not None:
            self.writer.add_video("video", animation, self.total_episodes)
