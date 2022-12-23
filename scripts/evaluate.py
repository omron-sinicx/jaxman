"""evaluate trained model

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import csv
import logging

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import hydra
import jax
import numpy as np
from jaxman.env.navigation.env import JaxMANEnv
from jaxman.planner.rl_planner.agent.sac.sac import create_sac_agent, restore_sac_actor
from jaxman.planner.rl_planner.rollout.rollout import build_rollout_episode


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


@hydra.main(config_path="config", config_name="eval_rl")
def main(config):
    logger = logging.getLogger("root")
    logger.addFilter(CheckTypesFilter())

    home_dir = "../../../.."

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
    actor = restore_sac_actor(
        actor, config.env.is_discrete, config.env.is_diff_drive, f"{home_dir}/model"
    )
    reward = []
    success = []
    makespan = []
    sum_of_cost = []
    rollout_fn = build_rollout_episode(env.instance, actor.apply_fn, evaluate=True)
    for i in range(config.eval_iters):
        # rollout episode
        key = jax.random.PRNGKey(i)
        carry = rollout_fn(key, actor.params, env.instance.obs)
        reward.append(carry.rewards.mean())
        trial_info = carry.trial_info
        success.append(trial_info.is_success)
        if trial_info.is_success:
            makespan.append(trial_info.makespan)
            sum_of_cost.append(trial_info.sum_of_cost)

    reward_mean = bs.bootstrap(np.array(reward), stat_func=bs_stats.mean)
    reward_std = bs.bootstrap(np.array(reward), stat_func=bs_stats.std)
    success_mean = bs.bootstrap(np.array(success), stat_func=bs_stats.std)
    success_std = bs.bootstrap(np.array(success), stat_func=bs_stats.std)
    makespan_mean = bs.bootstrap(np.array(makespan), stat_func=bs_stats.mean)
    makespan_std = bs.bootstrap(np.array(makespan), stat_func=bs_stats.std)
    sum_of_cost_mean = bs.bootstrap(np.array(sum_of_cost), stat_func=bs_stats.mean)
    sum_of_cost_std = bs.bootstrap(np.array(sum_of_cost), stat_func=bs_stats.std)

    # Path(f"{home_dir}/eval_data/{config.env.name}").mkdir(parents=True, exist_ok=True)
    # f = open(f"{home_dir}/eval_data/{config.env.name}/level{config.env.level}_num_agent{config.env.num_agents}/{config.model.name}.csv","w",)
    f = open("eval.cvs", "w")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["", "mean", "mean_lower", "mean_upper", "std"])
    csv_writer.writerow(
        [
            "reward",
            reward_mean.value,
            reward_mean.lower_bound,
            reward_mean.upper_bound,
            reward_std.value,
        ]
    )
    csv_writer.writerow(
        [
            "success",
            success_mean.value,
            success_mean.lower_bound,
            success_mean.upper_bound,
            success_std.value,
        ]
    )
    csv_writer.writerow(
        [
            "makespan",
            makespan_mean.value,
            makespan_mean.lower_bound,
            makespan_mean.upper_bound,
            makespan_std.value,
        ]
    )
    csv_writer.writerow(
        [
            "sum_of_cost",
            sum_of_cost_mean.value,
            sum_of_cost_mean.lower_bound,
            sum_of_cost_mean.upper_bound,
            sum_of_cost_std.value,
        ]
    )
    f.close()


if __name__ == "__main__":
    main()
