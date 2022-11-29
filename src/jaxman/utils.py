"""
utility functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import List

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


@jax.jit
def standardize(val):
    return (val - val.mean()) / (val.std() + 1e-10)
