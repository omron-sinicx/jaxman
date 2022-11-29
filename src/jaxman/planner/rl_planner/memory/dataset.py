"""Difinition of dataset for RL agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import NamedTuple, Union

import jax.numpy as jnp
import numpy as np
from chex import Array
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class Buffer:
    def __init__(
        self,
        observation_space: Box,
        action_space: Union[Box, Discrete],
        capacity: int = 10e7,
    ):
        """
        replay buffer

        Args:
            observation_space (Box): observation space
            action_space (Union[Box, Discrete]): action space
            capacity (int): buffer capacity
        """
        self.size = 0
        self.insert_index = 0
        self.observation_space = observation_space
        self.action_space = action_space

        # dim
        if isinstance(action_space, Discrete):
            self.act_dim = []
        else:
            self.act_dim = action_space.shape

        self.obs_dim = observation_space.shape[0]

        # buffer
        self.capacity = int(capacity)
        self.observations = np.zeros(
            (self.capacity, self.obs_dim), dtype=self.observation_space.dtype
        )
        self.actions = np.zeros(
            (self.capacity, *self.act_dim), dtype=self.action_space.dtype
        )
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.masks = np.zeros((self.capacity,), dtype=np.float32)
        self.dones_float = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, self.obs_dim), dtype=self.observation_space.dtype
        )


class TrainBatch(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    masks: Array
    next_observations: Array


class Experience(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    dones: Array

    @classmethod
    def reset(self, num_agents: int, T: int, observations: Array, actions: Array):
        """reset experience (make zeros tensor)

        Args:
            num_agents (int): number of agent
            T (int): maximum episode length
            observations (Array): agent observation
            actions (Array): agent action

        """
        observations = jnp.zeros([T + 1, *observations.shape])
        actions = jnp.zeros([T + 1, *actions.shape])
        rewards = jnp.zeros([T + 1, num_agents])
        dones = jnp.zeros([T + 1, num_agents])
        return self(observations, actions, rewards, dones)

    def push(
        self,
        idx: int,
        observations: Array,
        actions: Array,
        rewards: Array,
        dones: Array,
    ):
        """push agent experience to certain step

        Args:
            idx (int): inserting index
            observations (Array): agent observation
            actions (Array): agent action
            rewards (Array): reward
            dones (Array): done
        """
        observations = self.observations.at[idx].set(observations)
        actions = self.actions.at[idx].set(actions)
        rewards = self.rewards.at[idx].set(rewards)
        dones = self.dones.at[idx].set(dones)

        return Experience(
            observations,
            actions,
            rewards,
            dones,
        )
