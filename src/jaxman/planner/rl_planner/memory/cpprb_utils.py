"""utility functions for cpprb

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Union

import numpy as np
from cpprb import PrioritizedReplayBuffer, ReplayBuffer
from jaxman.utils import split_obs_and_comm

from .dataset import Experience, TrainBatch


def _push_experience_to_buffer(
    buffer: Union[ReplayBuffer, PrioritizedReplayBuffer], experience: Experience
):
    num_agents = experience.rewards.shape[-1]

    for i in range(num_agents):
        last_idx = np.where(experience.dones[:, i])[0][0]
        obs = np.copy(experience.observations[: last_idx + 1, i])
        act = np.copy(experience.actions[: last_idx + 1, i])
        next_obs = np.copy(experience.observations[1 : last_idx + 2, i])
        reward = np.copy(experience.rewards[: last_idx + 1, i])
        mask = np.copy(1 - experience.dones[: last_idx + 1, i])
        buffer.add(obs=obs, act=act, next_obs=next_obs, reward=reward, mask=mask)


def _sample_experience(
    buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    is_discrete: bool,
    use_per: bool,
    batch_size: int,
    num_agents: int,
    comm_dim: int,
    beta: float = 0.4,
):

    if use_per:
        sample_dict = buffer.sample(batch_size, beta)
        index = sample_dict["indexes"]
        weight = sample_dict["weights"]

    else:
        sample_dict = buffer.sample(batch_size)
        index = None
        weight = np.ones((batch_size,))

    all_obs = sample_dict["obs"]
    obss, comms, neighbor_masks = split_obs_and_comm(all_obs, num_agents, comm_dim)
    acts = sample_dict["act"]
    rews = sample_dict["reward"]
    masks = sample_dict["mask"]
    next_all_obs = sample_dict["next_obs"]
    next_obss, next_comms, next_neighbor_masks = split_obs_and_comm(
        next_all_obs, num_agents, comm_dim
    )

    rews = rews.flatten()
    masks = masks.flatten()
    if is_discrete:
        acts = acts.flatten().astype(int)

    data = TrainBatch(
        index,
        obss,
        comms,
        neighbor_masks,
        acts,
        rews,
        masks,
        next_obss,
        next_comms,
        next_neighbor_masks,
        weight,
    )

    return data
