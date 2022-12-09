"""Data structures for JaxMANEnv

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array

from .obstacle import ObstacleMap


class EnvInfo(NamedTuple):
    """Environmental information shared among agents and fixed for each problem instance"""

    num_agents: int
    occupancy_map: Optional[Array]
    sdf_map: Optional[Array]
    edges: Optional[Array]
    fov_r: int
    num_scans: int
    scan_range: float
    num_comm_agents: int
    timeout: int
    goal_reward: float
    crash_penalty: float
    time_penalty: float
    is_discrete: bool
    is_diff_drive: bool

    def is_valid(self) -> None:
        if self.occupancy_map is not None:
            chex.assert_rank(self.occupancy_map, 2)  # (width, height)
        if self.sdf_map is not None:
            chex.assert_rank(self.sdf_map, 2)  # (width, height)


class TrialInfo(NamedTuple):
    """Local planning information shared among agents and changed during each trial"""

    timesteps: Array = jnp.array(0)
    collided: Array = None
    obs_collided: Array = None
    agent_collided: Array = None
    collided_time: Array = None
    solved: Array = None
    solved_time: Array = None
    timeout: Array = None
    arrival_rate: Array = None
    crash_rate: Array = None
    sum_of_cost: Array = None
    makespan: Array = None
    is_success: Array = jnp.array(0).astype(bool)

    @classmethod
    def reset(self, num_agents: int):
        return self(
            timesteps=jnp.array(0),
            collided=jnp.zeros(num_agents).astype(bool),
            obs_collided=jnp.zeros(num_agents).astype(bool),
            agent_collided=jnp.zeros(num_agents).astype(bool),
            collided_time=jnp.ones(num_agents) * jnp.inf,
            solved=jnp.zeros(num_agents).astype(bool),
            solved_time=jnp.ones(num_agents) * jnp.inf,
            timeout=jnp.zeros(num_agents).astype(bool),
            arrival_rate=jnp.array(0),
            crash_rate=jnp.array(0),
            sum_of_cost=jnp.array(0),
            makespan=jnp.array(0),
            is_success=jnp.array(0).astype(bool),
        )


class TaskInfo(NamedTuple):
    """Local planning information shared among agents and fixed during each trial"""

    starts: Array
    start_rots: Array
    goals: Array
    obs: ObstacleMap

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack((self.starts, self.start_rots, self.goals))
        if as_numpy:
            ret = np.array(ret)
        return ret


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
        max_ang_vels = self.max_ang_vels[index]
        max_accs = self.max_accs[index]
        max_ang_accs = self.max_ang_accs[index]
        rads = self.rads[index]

        return self._replace(
            max_vels=max_vels,
            max_ang_vels=max_ang_vels,
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
        return ret

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


class AgentObservation(NamedTuple):
    """
    An observation generated by states, env_info, and task_info.
    Args:
        state (AgentState): Agents' states
        goal (Array): Agents' goal positions
        scans (Array): [n_agents, num_scans] array.
        planner_act (Array): DWA planner action suggestion
    """

    state: AgentState
    goals: Optional[Array] = None
    scans: Optional[Array] = None
    planner_act: Optional[Array] = None
    communications: Optional[Array] = None

    def cat(self, as_numpy=False) -> Array:
        """
        concatenate agent observations into a single array

        Args:
            as_numpy (bool, optional): whether to transform output to numpy array. Defaults to False (i.e., outputs jax array).

        Returns:
            Array: concatenated observation
        """
        ret_state = self.state.cat()
        ret = jnp.hstack(
            (
                ret_state,
                self.goals,
                self.scans,
            )
        )
        if self.planner_act is not None:
            ret = jnp.hstack((ret, self.planner_act))
        if self.communications is not None:
            num_agents = ret.shape[0]
            comm = self.communications.reshape([num_agents, -1])
            ret = jnp.hstack((ret, comm))

        if as_numpy:
            ret = np.array(ret)
        return ret

    def split_obs_comm(self) -> Tuple[Array, Array]:
        ret_state = self.state.cat()
        ret = jnp.hstack(
            (
                ret_state,
                self.goals,
                self.scans,
            )
        )
        if self.planner_act is not None:
            obs = jnp.hstack((ret, self.planner_act))
        else:
            obs = ret
        comm = self.communications

        return obs, comm

    def is_valid(self) -> None:
        # (n_agents, 2)
        chex.assert_rank(self.goals, 2)
