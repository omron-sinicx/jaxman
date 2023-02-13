"""Difinition of distributed woker (global buffer)

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import threading
import time
from typing import Union

import jax
import numpy as np
import ray
from gym.spaces import Box, Dict, Discrete
from omegaconf.dictconfig import DictConfig

from ..memory.dataset import Buffer, Experience, TrainBatch
from ..memory.utils import _build_push_experience_to_buffer, _build_sample_experience


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
            config.train.capacity,
        )

        self.batch_size = config.train.batch_size
        self.frame = 0
        self.batched_data = []

        self._push_experience_to_buffer = _build_push_experience_to_buffer(
            config.train.gamma,
            config.train.use_k_step_learning,
            config.train.k,
            config.env.timeout,
        )
        self._sample_train_batch = _build_sample_experience(config.train)

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
                data = self._sample_batch()
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_batched_data(self):
        """
        get one batch of data, called by learner.
        """

        if len(self.batched_data) == 0:
            data = self._sample_batch()
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
            self._push_experience_to_buffer(self, experience)
            del experience

    def _sample_batch(self) -> TrainBatch:
        """
        sample batched data

        Returns:
            TrainBatch: sampled batch data
        """
        with self.lock:
            self.key, data = self._sample_train_batch(
                self.key,
                self,
                self.num_agents,
                self.comm_dim,
                self.num_items,
                self.item_dim,
            )
            self.frame += 1
            return data

    def update_priority(self, index, priority):
        """
        add rollout episode experience to buffer

        Args:
            experience (Experience): rollout episode experience
        """
        with self.lock:
            np.put(self.priority, index, priority)
