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

from .dataset import Buffer, Experience, TrainBatch


def _build_push_experience_to_buffer(
    gamma: float, use_k_step_learning: bool, k: int, T: int
) -> Callable:
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

            buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
            buffer.insert_index = idx % buffer.capacity

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

            buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
            buffer.insert_index = idx % buffer.capacity

    if use_k_step_learning:
        return _push_k_step_learning_experience_to_buffer
    else:
        return _push_experience_to_buffer


def _build_sample_experience(
    train_config: DictConfig,
):
    batch_size = train_config.batch_size
    success_step_ratio = train_config.success_step_ratio

    def _build_sample_index():
        def uniform_sample(key: PRNGKey, size: int):
            index = np.random.randint(size, size=batch_size)
            return index, key

        return uniform_sample

    def _success_sample(buffer: Buffer, size: int):
        success_index = np.where(buffer.rewards > 0)[0]
        if len(success_index) > 0:
            index = np.random.randint(len(success_index), size=size)
            index = success_index[index]
        else:
            index = np.random.randint(buffer.size, size=size)
        return index

    _sample_index = _build_sample_index()

    def _sample_experience(
        key: PRNGKey,
        buffer: Buffer,
        num_agents: int,
        comm_dim: int,
        num_items: int,
        item_dim: int,
    ):
        index, key = _sample_index(key, buffer.size)
        if train_config.use_separete_sample:
            success_index = _success_sample(
                buffer, int(batch_size * success_step_ratio)
            )
            index = np.concatenate((index, success_index))

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

        data = TrainBatch(agent_obs, acts, rews, masks, next_agent_obs)

        return key, data

    return _sample_experience
