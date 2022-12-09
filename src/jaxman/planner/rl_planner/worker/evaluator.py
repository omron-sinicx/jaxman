"""Definition of Evaluator

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""


import threading
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ray
from flax.training.train_state import TrainState
from jaxman.env.core import TrialInfo
from jaxman.env.instance import Instance
from jaxman.env.viz import render_gif

from ..rollout.rollout import build_rollout_episode
from .learner import Learner


@ray.remote(num_cpus=1, num_gpus=0)
class Evaluator:
    def __init__(
        self,
        learner: Learner,
        actor: TrainState,
        instance: Instance,
        seed: int = 0,
    ) -> None:
        """
        Actor. collect agent rollout data

        Args:
            learner (Learner): learner. update agent parameters
            actor (TrainState): actor
            instance (Instance): envrironment instance
            seed (int, optional): seed. Defaults to 0
        """
        self.learner = learner
        self.actor = actor
        self.instance = instance

        self.seed = seed

        self.average_reward = jnp.zeros((self.instance.num_agents,))
        self.trial_info = []
        self.counter = 0
        self.last_counter = 0
        self.animation = None

        self._rollout_fn = build_rollout_episode(
            instance, actor.apply_fn, evaluate=False
        )

    def run(self) -> None:
        """
        remote run. collect agent rollout episode
        """
        self.learning_thread = threading.Thread(target=self._rollout, daemon=True)
        self.learning_thread.start()

    def _rollout(self) -> None:
        """
        collect agent rollout episode
        """
        key = jax.random.PRNGKey(self.seed)
        actor_params = self.actor.params

        while True:
            time.sleep(1)
            # rollout episode
            key, subkey = jax.random.split(key)
            carry = self._rollout_fn(subkey, actor_params, self.instance.obs)

            if self.counter % 20 == 0:
                # update parameters
                (
                    actor_params,
                    _,
                ) = self._update_parameters()

            self.average_reward += carry.rewards
            self.trial_info.append(carry.trial_info)

            # render episode trajectory
            if self.counter % 100 == 0:
                steps = carry.episode_steps
                state_traj = carry.experience.observations[:steps, :, :5]
                last_state = carry.state.cat()
                state_traj = jnp.vstack(
                    (state_traj, jnp.expand_dims(last_state, axis=0))
                )
                dones = carry.experience.dones[:steps, :]
                first_dones = jnp.zeros_like(dones[0], dtype=bool)
                dones = jnp.vstack((jnp.expand_dims(first_dones, 0), dones))
                animation = render_gif(
                    state_traj,
                    carry.task_info.goals,
                    self.instance.rads,
                    carry.task_info.obs.occupancy,
                    carry.trial_info,
                    dones,
                    self.instance.is_discrete,
                    self.instance.is_diff_drive,
                )
                self.animation = np.expand_dims(
                    np.transpose(np.array(animation), [0, 3, 1, 2]), 0
                )

            self.counter += 1

    def _update_parameters(self):
        """load actor parameters from learner"""
        actor_params_id, critic_params_id = ray.get(self.learner.get_params.remote())
        actor_params = ray.get(actor_params_id)
        critic_params = ray.get(critic_params_id)
        return actor_params, critic_params

    def stats(self, interval: int) -> Tuple[float, float, TrialInfo, np.ndarray]:
        """
        report current status of actor

        Args:
            interval (int): report interval

        Returns:
            Tuple[float, float, TrialInfo, np.ndarray]: average reward, average evaluation reward, trial info, rendering animation
        """
        num_rollouts = self.counter - self.last_counter
        print("number of eval rollout: {}".format(self.counter))
        print("eval rollout speed: {}/s".format(num_rollouts / interval))
        if num_rollouts > 0:
            print(
                "eval reward: {:.4f}".format(
                    np.mean(self.average_reward) / num_rollouts
                )
            )
            average_reward = self.average_reward.copy() / num_rollouts
            trial_info = self.trial_info.copy()
            animation = self.animation

            self.average_reward = jnp.zeros((self.instance.num_agents,))
            self.trial_info = []
            self.animation = None
        else:
            average_reward = trial_info = animation = None
        self.last_counter = self.counter
        return (
            average_reward,
            trial_info,
            animation,
        )
