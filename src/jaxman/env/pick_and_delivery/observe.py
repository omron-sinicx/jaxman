"""observation function for pick and delivery environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array

from ..base_observe import (
    _build_compute_neighbor_mask,
    _build_compute_relative_positions,
    _build_get_obs_pos,
    _build_get_other_agent_infos,
)
from ..core import AgentInfo, EnvInfo
from .core import AgentObservation, State, TaskInfo, TrialInfo


def _build_observe(env_info: EnvInfo, agent_info: AgentInfo) -> Callable:
    """build observe function

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information
        planner (DWAPlanner): DWA Planner

    Returns:
        Callable: jit-compiled observe function
    """
    num_items = env_info.num_items
    is_discrete = env_info.is_discrete
    _get_obs_pos = _build_get_obs_pos(env_info, agent_info)
    _get_other_agent_infos = _build_get_other_agent_infos(env_info)
    _get_mask = _build_compute_neighbor_mask(env_info)
    _get_item_mask = _build_compute_neighbor_item_mask(env_info)
    _compute_relative_pos = _build_compute_relative_positions(env_info)
    _compute_item_goals = _build_compute_item_goals(env_info)

    def _observe(state: State, task_info: TaskInfo, trial_info: TrialInfo):
        if is_discrete:
            obs_pos = _get_obs_pos(state, task_info.obs.occupancy)
        else:
            obs_pos = _get_obs_pos(state, task_info.obs.edges)

        is_hold_item = (state.load_item_id < num_items).reshape(-1, 1).astype(int)

        relative_positions = _get_other_agent_infos(
            state.agent_state, state.agent_state
        )

        intentions = jnp.zeros_like(relative_positions)

        relative_item_positions = _compute_relative_pos(
            state.agent_state, state.item_pos
        )
        item_info = jax.vmap(
            lambda a, b: jnp.concatenate((a, b), -1), in_axes=[0, None]
        )(relative_item_positions, task_info.item_goals)

        # compute mask
        not_finished_agents = jnp.logical_not(trial_info.agent_collided)
        agent_masks = _get_mask(relative_positions[:, :, :2], not_finished_agents)
        not_finished_items = jnp.logical_not(
            jnp.logical_or(trial_info.item_collided, trial_info.solved)
        )
        item_masks = _get_item_mask(
            relative_item_positions[:, :, :2], not_finished_items
        )

        item_goals = _compute_item_goals(state.load_item_id, task_info.item_goals)
        return AgentObservation(
            agent_state=state.agent_state,
            obs_scans=obs_pos,
            is_hold_item=is_hold_item,
            relative_positions=relative_positions,
            intentions=intentions,
            item_info=item_info,
            masks=agent_masks,
            item_masks=item_masks,
            item_goals=item_goals,
        )

    return jax.jit(_observe)


def _build_compute_item_goals(env_info):
    num_items = env_info.num_items

    def _inner_compute_item_goals(load_item_id: Array, item_goals: Array) -> Array:
        """
        Outputs the goal of the item the agent is carrying if the agent is carrying an item
        Outputs [0,0] if the agent isn’t carrying
        To be vmap

        Args:
            load_item_id (Array): ID of the item being carried by the agent.

        Returns:
            Array: item goals
        """
        is_carrying_item = load_item_id < num_items
        item_goal = item_goals[load_item_id]
        return item_goal * is_carrying_item

    def _compute_item_goals(load_item_id: Array, item_goals: Array) -> Array:
        return jax.vmap(_inner_compute_item_goals, in_axes=(0, None))(
            load_item_id, item_goals
        )

    return jax.jit(_compute_item_goals)


def _build_compute_neighbor_item_mask(
    env_info: EnvInfo,
):
    is_discrete = env_info.is_discrete
    r = env_info.comm_r

    def _compute_discrete_neighbor_mask(
        relative_pos: Array, not_finished_item: Array
    ) -> Array:
        """
        compute mask for obtaining only neighboring agent communications.
        neighbor is defined by distance between each agents.

        Args:
            relative_pos (Array): relative position between all agents
            not_finished_item (Array): array on whether each item completed the their episode.

        Returns:
            Array: mask specifying neighboring agents
        """
        agent_dist = jnp.max(jnp.abs(relative_pos), -1)
        neighbor_mask = agent_dist <= r
        neighbor_done_mask = jax.vmap(lambda a, b: a * b, in_axes=(0, None))(
            neighbor_mask, not_finished_item
        )
        return neighbor_done_mask

    def _compute_continuous_neighbor_mask(
        relative_pos: Array, not_finished_item: Array
    ) -> Array:
        """
        compute mask for obtaining only neighboring agent communications.
        neighbor is defined by distance between each agents.

        Args:
            relative_pos (Array): relative position between all agents
            not_finished_item (Array): array on whether each item completed the their episode.

        Returns:
            Array: mask specifying neighboring agents
        """
        agent_dist = jnp.sqrt(jnp.sum(relative_pos**2, axis=-1))
        neighbor_mask = agent_dist < r
        neighbor_done_mask = jax.vmap(lambda a, b: a * b, in_axes=(0, None))(
            neighbor_mask, not_finished_item
        )
        return neighbor_done_mask

    if is_discrete:
        return jax.jit(_compute_discrete_neighbor_mask)
    else:
        return jax.jit(_compute_continuous_neighbor_mask)


# def _build_get_obs_agent_item_map(env_info: EnvInfo, agent_info: AgentInfo) -> Callable:
#     def _build_get_discrete_obs_agent_item_map(env_info: EnvInfo):
#         fov_r = env_info.fov_r
#         num_agents = env_info.num_agents
#         _extract_fov = _build_extract_fov(env_info)
#         _add_agent_pos_to_obstacle_map = _build_add_agent_pos_to_obstacle_map(env_info)

#         def _get_obs_and_agent_pos(state: State, obs_map: Array) -> Array:
#             """
#             get flatten neighboring obstacles and agent position

#             Args:
#                 state (AgentState): agent's current state
#                 obs_map (Array): obstacle map. obs_map is added One padding

#             Returns:
#                 Array: flatten obs and agent position
#             """
#             obs_map = jnp.pad(obs_map, fov_r, mode="constant", constant_values=0)
#             agent_item_pos = jnp.vstack((state.agent_state.pos, state.item_pos))
#             obs_agent_map = _add_agent_pos_to_obstacle_map(agent_item_pos, obs_map)
#             fov = jax.vmap(_extract_fov, in_axes=(0, None))(
#                 state.agent_state, obs_agent_map
#             )
#             flatten_fov = fov.reshape(num_agents, -1)
#             return flatten_fov

#         return jax.jit(_get_obs_and_agent_pos)

#     def _build_get_continuous_obs_agent_item_map(env_info: EnvInfo):
#         pass

#     if env_info.is_discrete:
#         return _build_get_discrete_obs_agent_item_map(env_info)
#     else:
#         return _build_get_continuous_obs_agent_item_map(env_info)
