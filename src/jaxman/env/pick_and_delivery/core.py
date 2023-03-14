"""Data structures for Pick and Delivery Environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

from typing import NamedTuple

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
    agent_collided: Array = None
    item_collided: Array = None
    excess_move: Array = None
    solved: Array = None
    solved_time: Array = None
    timeout: Array = None
    delivery_rate: Array = None
    agent_crash_rate: Array = None
    item_crash_rate: Array = None
    sum_of_cost: Array = None
    makespan: Array = None
    is_success: Array = None

    @classmethod
    def reset(self, num_agents: int, num_items: int):
        return self(
            timesteps=jnp.array(0),
            agent_collided=jnp.zeros(num_agents).astype(bool),
            item_collided=jnp.zeros(num_items).astype(bool),
            excess_move=jnp.zeros(num_agents).astype(int),
            solved=jnp.zeros(num_items).astype(int),
            solved_time=jnp.ones(num_items) * jnp.inf,
            timeout=False,
            delivery_rate=jnp.array(0),
            agent_crash_rate=jnp.array(0),
            item_crash_rate=jnp.array(0),
            sum_of_cost=jnp.array(0),
            makespan=jnp.array(0),
            is_success=jnp.array(0).astype(bool),
        )


class TaskInfo(NamedTuple):
    """Information on pick and delivery tasks to solve"""

    starts: Array
    start_rots: Array
    item_starts: Array
    item_goals: Array
    obs: ObstacleMap

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack((self.starts, self.start_rots, self.goals))
        if as_numpy:
            ret = np.array(ret)
        return ret


class State(NamedTuple):
    """current agent and item state"""

    agent_state: AgentState = None
    load_item_id: Array = None
    life: Array = Array
    item_pos: Array = None
    item_time: Array = None

    def cat(self, as_numpy=False) -> Array:
        ret = jnp.hstack(
            (
                self.agent_state.pos,
                self.agent_state.rot,
                self.agent_state.vel,
                self.agent_state.ang,
                jnp.expand_dims(self.life, -1),
                jnp.expand_dims(self.load_item_id, -1),
            )
        )
        item_ret = jnp.hstack((self.item_pos, jnp.expand_dims(self.item_time, -1)))
        if as_numpy:
            ret = np.array(ret)
            item_ret = np.array(item_ret)
        return ret, item_ret

    @classmethod
    def from_array(self, agent_array: Array, item_array):
        agent_state = AgentState.from_array(agent_array[:, :-1])
        life = agent_array[:, -2]
        load_item_id = agent_array[:, -1]
        item_pos = item_array[:, :-1]
        item_item = item_array[:, -2]
        return State(agent_state, load_item_id, life, item_pos, item_item)


class AgentObservation(NamedTuple):
    """
    An agent partial observation
    Args:
        agent_state (AgentState): Agents' states
        obs_scans (Array): [n_agents, num_scans] array.
        life (Array): Agent's remaining life.
        is_hold_item (Array): whether an agent holding item or not
        relative_positions (Array): relative agent positions
        intentions (Array): next step agent greedy intention (where greedy agent want to move)
        hold_item_info: Array (Array): information of items held by other agent
        other_agent_life (Array): remaining life of other agents.
        item_info (Array): item relative positions and its goals
        masks (Array): mask to restrict agent communication to neighboring agents
        item_masks (Array): mask to restrict the item detection to neighboring item
        item_time (Array): Elapsed time since the item spawned
        item_goal (Array): carrying item goals. if agent is not carrying item, then item_goal set to (0,0)
    """

    agent_state: AgentState
    obs_scans: Array
    life: Array
    is_hold_item: Array
    relative_positions: Array
    intentions: Array
    hold_item_info: Array
    other_agent_life: Array
    item_info: Array
    masks: Array
    item_masks: Array
    item_time: Array
    item_goals: Array

    def cat(self, as_numpy=False) -> Array:
        """
        concatenate agent observations into a single array

        Args:
            as_numpy (bool, optional): whether to transform output to numpy array. Defaults to False (i.e., outputs jax array).

        Returns:
            Array: concatenated observation
        """
        ret_state = self.agent_state.cat()
        ret = jnp.hstack(
            (
                ret_state,
                self.obs_scans,
                self.life,
                self.is_hold_item,
                jnp.expand_dims(self.item_time, -1),
                self.item_goals,
            )
        )

        num_agents = ret.shape[0]

        # agent communication
        comm = jnp.concatenate(
            (
                self.relative_positions,
                self.intentions,
                self.other_agent_life,
                self.hold_item_info,
            ),
            axis=-1,
        )
        ret = jnp.hstack((ret, comm.reshape([num_agents, -1]), self.masks))

        # item communication
        ret = jnp.hstack(
            (ret, self.item_info.reshape([num_agents, -1]), self.item_masks)
        )

        if as_numpy:
            ret = np.array(ret)
        return ret

    def split_observation(self) -> ModelInput:
        """split agent observation into base_observation, communication, communication mask, item information, item mask

        Returns:
            ModelInput: formatted observation for flax model
        """
        ret_state = self.agent_state.cat()
        obs = jnp.hstack(
            (
                ret_state,
                self.obs_scans,
                self.life,
                self.is_hold_item,
                jnp.expand_dims(self.item_time, -1),
                self.item_goals,
            )
        )
        comm = jnp.concatenate(
            (
                self.relative_positions,
                self.intentions,
                self.other_agent_life,
                self.hold_item_info,
            ),
            axis=-1,
        )
        mask = self.masks

        item_pos = self.item_info
        item_mask = self.item_masks

        return ModelInput(obs, comm, mask, item_pos, item_mask, self.is_hold_item)

    def is_valid(self) -> None:
        # (n_agents, 2)
        chex.assert_rank(self.goals, 2)
