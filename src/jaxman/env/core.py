"""Data structures for JaxMANEnv

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array


class EnvInfo(NamedTuple):
    """Environmental information shared among agents and fixed for each problem instance"""

    env_name: str
    num_agents: int
    num_items: int
    max_life: int
    occupancy_map: Optional[Array]
    sdf_map: Optional[Array]
    edges: Optional[Array]
    item_rads: Array
    fov_r: int
    comm_r: float
    num_scans: int
    scan_range: float
    use_intentions: bool
    use_hold_item_info: bool
    timeout: int
    is_crashable: bool
    is_biased_sample: bool
    is_respawn: bool
    goal_reward: float
    dist_reward: float
    dont_hold_item_penalty: float
    crash_penalty: float
    time_penalty: float
    pickup_reward: float
    life_penalty: float
    is_decay_reward: bool
    decay_start: int
    decay_end: int
    min_reward: float
    is_discrete: bool
    is_diff_drive: bool

    def is_valid(self) -> None:
        if self.occupancy_map is not None:
            chex.assert_rank(self.occupancy_map, 2)  # (width, height)
        if self.sdf_map is not None:
            chex.assert_rank(self.sdf_map, 2)  # (width, height)


class AgentInfo(NamedTuple):
    """Kinematic information given to each individual agent and fixed for each problem instance"""

    max_vels: Array
    min_vels: Array
    max_ang_vels: Array
    min_ang_vels: Array
    max_accs: Array
    max_ang_accs: Array
    rads: Array

    @jax.jit
    def at(self, index: int) -> AgentInfo:
        """
        extract info for the agent of selected index

        Args:
            index (int): agent index

        Returns:
            AgentInfo: info for the specified agent
        """
        max_vels = self.max_vels[index]
        min_vels = self.min_vels[index]
        max_ang_vels = self.max_ang_vels[index]
        min_ang_vels = self.min_ang_vels[index]
        max_accs = self.max_accs[index]
        max_ang_accs = self.max_ang_accs[index]
        rads = self.rads[index]

        return self._replace(
            max_vels=max_vels,
            min_vels=min_vels,
            max_ang_vels=max_ang_vels,
            min_ang_vels=min_ang_vels,
            max_accs=max_accs,
            max_ang_accs=max_ang_accs,
            rads=rads,
        )


class AgentState(NamedTuple):
    """
    Args:
        pos (Array): [n_agents, 2]-array
        rot (Array): [n_agents, 1]-array
        vel (Array): [n_agents, 1]-array
        ang (Array): [n_agents, 1]-array
    """

    pos: Optional[Array] = None
    rot: Optional[Array] = None
    vel: Optional[Array] = None
    ang: Optional[Array] = None

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack((self.pos, self.rot, self.vel, self.ang))
        if as_numpy:
            ret = np.array(ret)
        return ret.astype(self.pos.dtype)

    def is_valid(self) -> None:
        chex.assert_shape(self.pos, (..., 2))
        chex.assert_shape(self.rot, (..., 1))
        chex.assert_shape(self.vel, (..., 1))
        chex.assert_shape(self.ang, (..., 1))

    @classmethod
    def from_array(self, array: Array):
        pos = array[..., :2]
        rot = array[..., 2:3]  # (..., 1)
        vel = array[..., 3:4]  # (..., 1)
        ang = array[..., 4:5]  # (..., 1)
        state = self(pos=pos, rot=rot, vel=vel, ang=ang)
        state.is_valid()
        return state

    @classmethod
    def init(self, num_agents: int):
        pos = jnp.zeros((num_agents, 2))
        rot = jnp.zeros((num_agents, 1))
        vel = jnp.zeros((num_agents, 1))
        ang = jnp.zeros((num_agents, 1))
        return self(pos, rot, vel, ang)


class AgentAction(NamedTuple):
    """
    Args:
        normed_vel (Array): [n_agents, 1]-array
        normed_ang (Array): [n_agents, 1]-array
    """

    normed_vel: float
    normed_ang: float

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack((self.normed_vel, self.normed_ang))
        if as_numpy:
            ret = np.array(ret)
        return ret

    def is_valid(self) -> None:
        chex.assert_equal_rank((self.normed_vel, self.normed_ang))
        chex.assert_shape((self.normed_vel, self.normed_ang), (..., 1))

    @classmethod
    def from_array(self, array: Array):
        normed_vel = array[..., 0:1]  # (..., 1)
        normed_ang = array[..., 1:2]  # (..., 1)
        act = self(normed_vel=normed_vel, normed_ang=normed_ang)
        act.is_valid()
        return act
