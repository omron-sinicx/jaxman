"""
utility functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import List, Tuple

import jax
import jax.numpy as jnp
from chex import Array

ACTION_DICT = {
    "STAY": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "DOWN": 3,
    "UP": 4,
    "GO": 1,
    "BACK": 2,
    "TURN_L": 3,
    "TURN_R": 4,
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


def split_obs_and_comm(
    observations: Array, num_agents: int, comm_dim: int
) -> Tuple[Array, Array]:
    """split observation into agent basic observations and communications

    Args:
        observations (Array): observations, contrain basic obs and comm
        num_agents (int): number of agent in whole environment
        comm_dim (int): communication dimensions

    Returns:
        Tuple[Array, Array]: observations, communications
    """

    batch_size = observations.shape[0]
    total_comm_dim = num_agents * comm_dim
    mask_dim = num_agents
    obs = observations[:, : -total_comm_dim - mask_dim]
    comm = observations[:, -total_comm_dim - mask_dim : -mask_dim].reshape(
        batch_size, num_agents, comm_dim
    )
    mask = observations[:, -mask_dim:]
    return obs, comm, mask
