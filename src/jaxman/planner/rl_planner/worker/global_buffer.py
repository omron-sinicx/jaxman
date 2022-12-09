"""Difinition of distributed woker (global buffer)

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import threading
import time
from typing import Union

import jax
import ray
from gym.spaces import Box, Dict, Discrete
from omegaconf.dictconfig import DictConfig

from ..memory.dataset import Buffer, Experience, TrainBatch
from ..memory.utils import _push_experience_to_buffer, _sample_experience


@ray.remote(num_cpus=1, num_gpus=0)
class GlobalBuffer(Buffer):
    def __init__(
        self,
        observation_space: Dict,
        action_space: Union[Discrete, Box],
        config: DictConfig,
    ):
        """
        replay buffer

        Args:
            observations_space (Dict): observation space
            actions (Union[Discrete, Box]): action space
            config (DictConfig): configuration
        """
        super().__init__(
            observation_space,
            action_space,
            config.capacity,
        )

        self.batch_size = config.batch_size
        self.frame = 0
        self.batched_data = []

        self.key = jax.random.PRNGKey(0)
        self.lock = threading.Lock()

    def num_data(self) -> int:
        """
        return buffer size. called by another ray actor

        Returns:
            int: buffer size
        """
        size = self.size
        size_id = ray.put(size)
        return size_id

    def run(self):
        """
        prepare data
        """
        self.background_thread = threading.Thread(
            target=self._prepare_data, daemon=True
        )
        self.background_thread.start()

    def _prepare_data(self):
        """
        prepare batched data for training
        """
        while True:
            if len(self.batched_data) <= 10:
                data = self._sample_batch(self.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_batched_data(self):
        """
        get one batch of data, called by learner.
        """

        if len(self.batched_data) == 0:
            data = self._sample_batch(self.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, experience: Experience):
        """
        add rollout episode experience to buffer

        Args:
            experience (Experience): rollout episode experience
        """
        with self.lock:
            _push_experience_to_buffer(self, experience)
            del experience

    def _sample_batch(self, batch_size: int) -> TrainBatch:
        """
        sample batched data

        Args:
            batch_size (int): size of batch

        Returns:
            TrainBatch: sampled batch data
        """
        with self.lock:
            data = _sample_experience(
                self, batch_size, self.num_comm_agents, self.comm_dim
            )
            self.frame += 1
            return data
