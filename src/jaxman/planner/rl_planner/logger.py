"""Definition of logger, log training and evaluation data

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import jax.numpy as jnp
import numpy as np

# @jax.jit
# def calc_var(values: Array):
#     num_agents = values.shape[0]
#     mean = values.mean()
#     cov = jnp.sum((values - mean) ** 2)
#     cov = cov / (num_agents - 1)
#     cov *= ~jnp.isinf(cov)
#     return cov


# @jax.jit
# def calc_ggf(values: Array, rate: int = 0.5):
#     num_agents = values.shape[0]
#     # weight is Monotonically decreasing geometric progression
#     weight = jnp.ones(num_agents) * rate
#     weight = weight ** jnp.arange(num_agents)
#     # normalized weight
#     weight = weight / weight.sum()
#     # values are sorted in an increasing order
#     values = jnp.sort(values)
#     ggf = jnp.sum(values * weight)
#     return ggf


class LogResult:
    def __init__(self, writer, config) -> None:
        self.total_episodes = 0
        self.total_updates = 0
        self.writer = writer
        # Path("./data").mkdir(parents=True, exist_ok=True)
        # self.f = open(f"./data/{config.seed}.csv","w",)
        # self.csv_writer = csv.writer(self.f)
        # self.csv_writer.writerow(
        #     [
        #         "step",
        #         "reward",
        #         "sr",
        #         "cov",
        #         "mean",
        #         "max",
        #         "mkspan",
        #         "soc",
        #         "sw_cov",
        #         "sw_mean",
        #         "sw_max",
        #         "sw_mkspan",
        #         "sw_soc",
        #     ]
        # )

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
                # collided = sum(trial_info[i].collided)
                # self.writer.add_scalar(
                #     "evaluation/collided", collided, self.total_episodes
                # )
                solved = sum(trial_info[i].solved)
                self.writer.add_scalar("evaluation/solved", solved, self.total_episodes)
                # timeout = sum(trial_info[i].timeout)
                # self.writer.add_scalar(
                #     "evaluation/timeout", timeout, self.total_episodes
                # )
                is_success = trial_info[i].is_success
                self.writer.add_scalar(
                    "evaluation/is_success",
                    is_success.astype(float),
                    self.total_episodes,
                )
                if is_success:
                    success = 1
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

            # self.csv_writer.writerow(
            #     [
            #         self.total_episodes,
            #         jnp.mean(eval_reward[i]),
            #         success,
            #         delay_cov,
            #         delay_mean,
            #         delay_max,
            #         makespan,
            #         sum_of_cost,
            #         sw_cov,
            #         sw_mean,
            #         sw_max,
            #         sw_mk,
            #         sw_soc,
            #     ]
            # )

        if animation is not None:
            self.writer.add_video("video", animation, self.total_episodes)

    # def close(self):
    #     self.f.close()
