"""Difinition of dataset for RL agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import NamedTuple, Union

import jax.numpy as jnp
import numpy as np
from chex import Array
from gym.spaces import Box, Dict, Discrete


class Buffer:
    def __init__(
        self,
        observation_space: Dict,
        action_space: Union[Discrete, Box],
        capacity: int = 10e7,
    ):
        """
        replay buffer

        Args:
            observation_space (Dict): observation space
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

        self.obs_dim = observation_space["obs"].shape[0]
        self.num_agents = observation_space["comm"].shape[0]
        self.comm_dim = observation_space["comm"].shape[1]
        self.mask_dim = observation_space["mask"].shape[0]

        total_obs_dim = self.obs_dim + self.num_agents * self.comm_dim + self.mask_dim
        # buffer
        self.capacity = int(capacity)
        self.observations = np.zeros(
            (self.capacity, total_obs_dim), dtype=observation_space["obs"].dtype
        )
        self.actions = np.zeros(
            (self.capacity, *self.act_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.masks = np.zeros((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros(
            (self.capacity, total_obs_dim), dtype=observation_space["obs"].dtype
        )


class TrainBatch(NamedTuple):
    observations: Array
    communications: Array
    neighbor_masks: Array
    actions: Array
    rewards: Array
    masks: Array
    next_observations: Array
    next_communications: Array
    next_neighbor_masks: Array


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
