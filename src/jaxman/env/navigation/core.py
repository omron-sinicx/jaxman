"""Data structures for Navigation Environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import chex
import jax.numpy as jnp
import numpy as np
from chex import Array
from jaxman.planner.rl_planner.core import AgentObservation as ModelInput

from ..core import AgentState
from ..obstacle import ObstacleMap


class TrialInfo(NamedTuple):
    """current trial status updated every step"""

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
    """Information on navigation tasks to solve"""

    starts: Array
    start_rots: Array
    goals: Array
    obs: ObstacleMap

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack((self.starts, self.start_rots, self.goals))
        if as_numpy:
            ret = np.array(ret)
        return ret


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
    An agent partial observation
    Args:
        state (AgentState): Agents' states
        goals (Array): Agents' goal positions
        scans (Array): [n_agents, num_scans] array.
        planner_act (Array): DWA planner action suggestion
        relative_positions (Array): relative agent positions
        intentions (Array): next step agent greedy intention (where greedy agent want to move)
        masks (Array): mask to restrict agent communication to neighboring agents
        extra_obs (Optional[Array]): extra observation e.g., comulative cost, reward, etc. this will be concatenated to the observation
        extra_comm (Optional[Array]): extra communication e.g., priority of other agents, etc. this will be concatenated to the communication
    """

    state: AgentState
    goals: Optional[Array] = None
    scans: Optional[Array] = None
    planner_act: Optional[Array] = None
    relative_positions: Optional[Array] = None
    intentions: Optional[Array] = None
    masks: Optional[Array] = None
    extra_obs: Optional[Array] = None
    extra_comm: Optional[Array] = None

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
        if self.extra_obs is not None:
            ret = jnp.hstack((ret, self.extra_obs))
        if self.planner_act is not None:
            ret = jnp.hstack((ret, self.planner_act))
        if self.relative_positions is not None:
            comm = jnp.concatenate((self.relative_positions, self.intentions), axis=-1)
            num_agents = ret.shape[0]
            if self.extra_comm is not None:
                comm = jnp.concatenate((comm, self.extra_comm), axis=-1)
        ret = jnp.hstack((ret, comm.reshape([num_agents, -1])))
        
        if self.masks is not None:
            ret = jnp.hstack((ret, self.masks))

        if as_numpy:
            ret = np.array(ret)
        return ret

    def split_observation(self) -> Tuple[Array, Array, Array]:
        """split agent observation into base_observation, communication and communication_mask

        Returns:
            Tuple[Array, Array, Array]: base_observation, communication, communication_mask
        """
        ret_state = self.state.cat()
        obs = jnp.hstack(
            (
                ret_state,
                self.goals,
                self.scans,
            )
        )
        if self.extra_obs is not None:
            obs = jnp.hstack((obs, self.extra_obs))
        if self.planner_act is not None:
            obs = jnp.hstack((obs, self.planner_act))
        
        comm = jnp.concatenate((self.relative_positions, self.intentions), axis=-1)
        if self.extra_comm is not None:
            comm = jnp.concatenate((comm, self.extra_comm), axis=-1)
        mask = self.masks

        return ModelInput(obs, comm, mask)

    def is_valid(self) -> None:
        # (n_agents, 2)
        chex.assert_rank(self.goals, 2)
