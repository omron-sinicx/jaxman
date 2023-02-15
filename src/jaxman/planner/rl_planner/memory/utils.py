"""utility functions for memory

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from jaxman.utils import split_obs_and_comm
from omegaconf.dictconfig import DictConfig

from .dataset import Buffer, Experience, PERTrainBatch


def _build_push_experience_to_buffer(
    gamma: float, use_k_step_learning: bool, k: int, T: int
) -> Callable:
    def _build_apply_gamma(gamma: float, k: int):
        weight = jnp.array([gamma**i for i in range(k)])
        index = jnp.arange(T)

        # to be vmap
        def _apply_gamma(rewards: Array, index: Array):
            rews = jax.lax.dynamic_slice_in_dim(rewards, index, k)
            return jnp.sum(rews * weight)

        def apply_gamma(rewards: Array):
            rews = jax.vmap(_apply_gamma, in_axes=(None, 0))(rewards, index)
            return rews

        return jax.jit(apply_gamma)

    apply_gamma = _build_apply_gamma(gamma, k)

    def _push_experience_to_buffer(buffer: Buffer, experience: Experience):
        idx = buffer.insert_index
        num_agents = experience.rewards.shape[-1]

        for i in range(num_agents):
            last_idx = np.where(experience.dones[:, i])[0][0]
            if idx + last_idx + 1 > buffer.capacity:
                extra_length = (idx + last_idx + 1) - buffer.capacity
                last_idx = last_idx - extra_length
                buffer.size = buffer.capacity

            buffer.observations[idx : idx + last_idx + 1] = np.copy(
                experience.observations[: last_idx + 1, i]
            )
            buffer.actions[idx : idx + last_idx + 1] = np.copy(
                experience.actions[: last_idx + 1, i]
            )
            buffer.rewards[idx : idx + last_idx + 1] = np.copy(
                experience.rewards[: last_idx + 1, i]
            )
            buffer.masks[idx : idx + last_idx + 1] = np.copy(
                1 - experience.dones[: last_idx + 1, i]
            )
            buffer.next_observations[idx : idx + last_idx + 1] = np.copy(
                experience.observations[1 : last_idx + 2, i]
            )
            idx += last_idx + 1
            # PER
            if buffer.size == 0:
                buffer.priority[idx : idx + last_idx + 1] = 1
            else:
                prio = np.max(buffer.priority)
                buffer.priority[idx : idx + last_idx + 1] = prio

            buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
            buffer.insert_index = idx % buffer.capacity

    def _push_k_step_learning_experience_to_buffer(
        buffer: Buffer, experience: Experience
    ):
        idx = buffer.insert_index
        num_agents = experience.rewards.shape[-1]

        dummy_obs = jnp.tile(jnp.zeros_like(experience.observations[0][0]), k).reshape(
            k, -1
        )
        dummy_rew = jnp.zeros((k,))
        dummy_done = jnp.ones((k,))

        for i in range(num_agents):
            last_idx = np.where(experience.dones[:, i])[0][0]
            if idx + last_idx + 1 > buffer.capacity:
                extra_length = (idx + last_idx + 1) - buffer.capacity
                last_idx = last_idx - extra_length
                buffer.size = buffer.capacity

            buffer.observations[idx : idx + last_idx + 1] = np.copy(
                experience.observations[: last_idx + 1, i]
            )
            buffer.actions[idx : idx + last_idx + 1] = np.copy(
                experience.actions[: last_idx + 1, i]
            )

            # for multi step learning
            dones = jnp.concatenate((experience.dones[: last_idx + 1, i], dummy_done))
            buffer.masks[idx : idx + last_idx + 1] = np.copy(
                1 - dones[k - 1 : last_idx + k]
            )
            observations = jnp.concatenate(
                (experience.observations[: last_idx + 1, i], dummy_obs)
            )
            buffer.next_observations[idx : idx + last_idx + 1] = np.copy(
                observations[k : last_idx + 1 + k]
            )

            # apply decay rate
            rewards = jnp.concatenate(
                (experience.rewards[: last_idx + 1, i], dummy_rew)
            )
            rewards = apply_gamma(rewards)
            buffer.rewards[idx : idx + last_idx + 1] = np.copy(rewards[: last_idx + 1])
            idx += last_idx + 1
            # PER
            if buffer.size == 0:
                buffer.priority[idx : idx + last_idx + 1] = 1
            else:
                prio = np.max(buffer.priority)
                buffer.priority[idx : idx + last_idx + 1] = prio

            buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
            buffer.insert_index = idx % buffer.capacity

    if use_k_step_learning:
        return _push_k_step_learning_experience_to_buffer
    else:
        return _push_experience_to_buffer


def _build_sample_experience(
    train_config: DictConfig,
):
    def _build_sample_index(
        use_per: bool,
    ):
        buffer_index = jnp.arange(train_config.capacity)

        @jax.jit
        def per_sample(
            key: PRNGKey,
            priority: Array,
            size: int,
        ):
            key, subkey = jax.random.split(key)
            p = priority + 1e-8
            p = (p * (buffer_index < size)) ** train_config.per_alpha
            probs = p / jnp.sum(p)
            index = jax.random.choice(
                subkey, train_config.capacity, shape=(train_config.batch_size,), p=probs
            )
            probs = jax.vmap(lambda probs, idx: probs[idx], in_axes=(None, 0))(
                probs, index
            )
            weight = ((1 / size) * (1 / probs)) ** train_config.per_beta
            return index, weight, key

        def uniform_sample(key: PRNGKey, _, size: int):
            index = np.random.randint(size, size=train_config.batch_size)
            weight = np.ones((train_config.batch_size,))
            return index, weight, key

        if use_per:
            return per_sample
        else:
            return uniform_sample

    def _batch_normalization(buffer: Buffer, sample_reward: np.array) -> np.array:
        """normalized reward and update reward mean and std

        Args:
            buffer (Buffer): global buffer
            sample_reward (np.array): rewards sampled as mini-batch

        Returns:
            np.array: standardized reward
        """
        if train_config.use_batch_norm:
            mean = np.mean(sample_reward)
            std = np.std(sample_reward) + 1e-8
            if buffer.reward_mean is not None:
                mean = buffer.reward_mean * train_config.batch_norm_tau + mean * (
                    1 - train_config.batch_norm_tau
                )
                std = buffer.reward_var * train_config.batch_norm_tau + std * (
                    1 - train_config.batch_norm_tau
                )
            buffer.reward_mean = mean
            buffer.reward_var = std
            if train_config.use_only_mean:
                sample_reward = sample_reward - mean
            if train_config.use_only_std:
                sample_reward = sample_reward / std
            else:
                sample_reward = (sample_reward - mean) / std

        return sample_reward

    _sample_index = _build_sample_index(train_config.use_per)

    def _sample_experience(
        key: PRNGKey,
        buffer: Buffer,
        num_agents: int,
        comm_dim: int,
        num_items: int,
        item_dim: int,
    ):
        index, weight, key = _sample_index(key, buffer.priority, buffer.size)

        all_obs = buffer.observations[index]
        agent_obs = split_obs_and_comm(
            all_obs, num_agents, comm_dim, num_items, item_dim
        )
        acts = buffer.actions[index]
        rews = buffer.rewards[index]
        masks = buffer.masks[index]
        next_all_obs = buffer.next_observations[index]
        next_agent_obs = split_obs_and_comm(
            next_all_obs, num_agents, comm_dim, num_items, item_dim
        )

        rews = _batch_normalization(buffer, rews)

        data = PERTrainBatch(
            index, agent_obs, acts, rews, masks, next_agent_obs, weight
        )

        return key, data

    return _sample_experience
