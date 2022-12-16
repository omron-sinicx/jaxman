"""utility functions for memory

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
import numpy as np
from jaxman.utils import split_obs_and_comm

from .dataset import Buffer, Experience, TrainBatch


def _push_experience_to_buffer(buffer: Buffer, experience: Experience):
    idx = buffer.insert_index
    num_agents = experience.rewards.shape[-1]

    for i in range(num_agents):
        last_idx = np.where(experience.dones[:, i])[0][0]
        if idx + last_idx + 1 > buffer.capacity:
            idx = 0
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
        buffer.insert_index = idx


def _sample_experience(buffer: Buffer, batch_size: int, num_agents: int, comm_dim: int):
    index = np.random.randint(buffer.size, size=batch_size)

    all_obs = buffer.observations[index]
    obss, comms, neighbor_masks = split_obs_and_comm(all_obs, num_agents, comm_dim)
    acts = buffer.actions[index]
    rews = buffer.rewards[index]
    masks = buffer.masks[index]
    next_all_obs = buffer.next_observations[index]
    next_obss, next_comms, next_neighbor_masks = split_obs_and_comm(
        next_all_obs, num_agents, comm_dim
    )

    data = TrainBatch(
        obss,
        comms,
        neighbor_masks,
        acts,
        rews,
        masks,
        next_obss,
        next_comms,
        next_neighbor_masks,
    )

    return data
