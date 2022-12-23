"""
utility functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import List, Tuple

import jax
import jax.numpy as jnp
from chex import Array

from .planner.rl_planner.core import AgentObservation

ACTION_DICT = {
    "STAY": 0,
    "RIGHT": 1,
    "LEFT": 2,
    "DOWN": 3,
    "UP": 4,
    "GO": 1,
    "BACK": 2,
    "TURN_L": 3,
    "TURN_R": 4,
    "LOAD": 5,
    "UNLOAD": 5,
}


@jax.jit
def value_at_i(value: Array, i: int):
    return value[i]


def compute_agent_action(action: List):
    actions = []
    for i in range(len(action)):
        actions.append(ACTION_DICT[action[i]])
    return jnp.array(actions)


def standardize(val):
    return (val - val.mean()) / (val.std() + 1e-10)


@jax.jit
def compute_distance(a: Array, b: Array):
    return jnp.sqrt(jnp.sum((a - b) ** 2))


def split_obs_and_comm(
    observations: Array,
    num_agents: int,
    comm_dim: int,
    num_items: int,
    item_dim: int,
) -> Tuple[Array, Array]:
    """split observation into agent basic observations and communications

    Args:
        observations (Array): observations, contrain basic obs and comm
        num_agents (int): number of agent in whole environment
        comm_dim (int): communication dimensions
        num_items (int): number of items in environment
        item_dim (int): dimension of item information

    Returns:
        AgentObservation: NamedTuple of agent observations
    """

    batch_size = observations.shape[0]
    total_comm_dim = num_agents * comm_dim
    mask_dim = num_agents
    total_item_dim = num_items * item_dim
    item_mask_dim = num_items
    obs = observations[:, : -total_comm_dim - mask_dim - total_item_dim - item_mask_dim]
    comm = observations[
        :,
        -total_comm_dim
        - mask_dim
        - total_item_dim
        - item_mask_dim : -mask_dim
        - total_item_dim
        - item_mask_dim,
    ].reshape(batch_size, num_agents, comm_dim)
    if num_items > 0:
        mask = observations[
            :,
            -mask_dim
            - total_item_dim
            - item_mask_dim : -total_item_dim
            - item_mask_dim,
        ]
        item_pos = observations[
            :, -total_item_dim - item_mask_dim : -item_mask_dim
        ].reshape(batch_size, num_items, item_dim)
        item_mask = observations[:, -item_mask_dim:]
    else:
        mask = observations[:, -mask_dim:]
        item_pos = None
        item_mask = None
    # print("obs",obs)
    # print("comm",comm)
    # print("mask",mask)
    # print("item_pos",item_pos)
    # print("item_mask",item_mask)
    # print()

    return AgentObservation(obs, comm, mask, item_pos, item_mask)
