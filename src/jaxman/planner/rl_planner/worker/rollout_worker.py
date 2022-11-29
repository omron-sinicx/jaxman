"""Definition of rollout worker (collect training data)

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

from ..rollout.rollout import build_rollout_episode
from .global_buffer import GlobalBuffer
from .learner import Learner


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(
        self,
        global_buffer: GlobalBuffer,
        learner: Learner,
        actor: TrainState,
        instance: Instance,
        seed: int = 0,
    ) -> None:
        """
        Actor. collect agent rollout data

        Args:
            global_buffer (GlobalBuffer): global buffer
            learner (Learner): learner. update agent parameters
            actor (TrainState): actor
            instance (Instance): envrironment instance
            seed (int, optional): seed. Defaults to 0
        """
        self.global_buffer = global_buffer
        self.learner = learner
        self.actor = actor
        self.instance = instance

        self.seed = seed

        self.average_reward = jnp.zeros((self.instance.num_agents,))
        self.counter = 0
        self.last_counter = 0

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
            # rollout episode
            time.sleep(0.1)
            key, subkey = jax.random.split(key)
            carry = self._rollout_fn(subkey, actor_params, self.instance.obs)

            experience = carry.experience
            self.global_buffer.add.remote(experience)

            if self.counter % 20 == 0:
                # update parameters
                (
                    actor_params,
                    _,
                ) = self._update_parameters()

            self.average_reward += carry.rewards

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
        print("number of rollout: {}".format(self.counter))
        print(
            "rollout speed: {}/s".format((self.counter - self.last_counter) / interval)
        )
        if self.counter != self.last_counter:
            average_reward = self.average_reward / (self.counter - self.last_counter)
            print("reward: {:.4f}".format(jnp.mean(average_reward)))
        else:
            average_reward = None
        self.last_counter = self.counter
        self.average_reward = jnp.zeros((self.instance.num_agents,))
        self.average_meta_reward = jnp.zeros((self.instance.num_agents,))

        return average_reward
