"""observation function for pick and delivery environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array

from ...planner.dwa import create_planner
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

    Returns:
        Callable: jit-compiled observe function
    """
    num_agents = env_info.num_agents
    num_items = env_info.num_items
    is_discrete = env_info.is_discrete
    _get_obs_pos = _build_get_obs_pos(env_info, agent_info)
    _get_other_agent_infos = _build_get_other_agent_infos(env_info)
    _get_mask = _build_compute_neighbor_mask(env_info)
    _get_item_mask = _build_compute_neighbor_item_mask(env_info)
    _compute_relative_pos = _build_compute_relative_positions(env_info)
    _compute_item_time = _build_compute_item_times(env_info)
    _compute_item_goals = _build_compute_item_goals(env_info)
    _compute_hold_item_info = _build_compute_hold_item_info(env_info)
    _planner = create_planner(env_info, agent_info)

    def _observe(
        state: State, task_info: TaskInfo, trial_info: TrialInfo
    ) -> AgentObservation:
        """observe

        Args:
            state (State): current agent and item state
            task_info (TaskInfo): task information
            trial_info (TrialInfo): trial information

        Returns:
            AgentObservation: observation
        """
        if is_discrete:
            obs_pos = _get_obs_pos(state, task_info.obs.occupancy)
        else:
            obs_pos = _get_obs_pos(state, task_info.obs.edges)

        life = state.life.reshape(-1, 1)
        is_hold_item = (state.load_item_id < num_items).reshape(-1, 1).astype(int)
        item_time = _compute_item_time(state.load_item_id, state.item_time)
        item_starts = _compute_item_goals(state.load_item_id, task_info.item_starts)
        item_goals = _compute_item_goals(state.load_item_id, task_info.item_goals)

        # other agent communication
        relative_positions = _get_other_agent_infos(
            state.agent_state, state.agent_state
        )
        intentions = jnp.zeros_like(relative_positions)
        other_agent_life = jnp.tile(life, num_agents).reshape(num_agents, num_agents, 1)
        hold_item_info = _compute_hold_item_info(is_hold_item, item_goals, item_time)

        # item information
        relative_item_positions = _compute_relative_pos(
            state.agent_state, state.item_pos
        )
        item_info = jax.vmap(
            lambda a, b, c: jnp.concatenate((a, b, c), -1), in_axes=[0, None, None]
        )(
            relative_item_positions,
            task_info.item_goals + jnp.array([0, 1]),
            jnp.expand_dims(state.item_time, -1),
        )

        # compute mask
        not_finished_agents = jnp.logical_not(trial_info.agent_collided)
        agent_masks = _get_mask(relative_positions[:, :, :2], not_finished_agents)
        not_finished_items = jnp.logical_not(
            jnp.logical_or(trial_info.item_collided, trial_info.solved)
        )
        item_masks = _get_item_mask(
            relative_item_positions[:, :, :2], not_finished_items
        )

        # Classical planner action
        if is_discrete:
            planner_act = None
        else:
            planner_act = (
                _planner._act(state.agent_state, item_goals, task_info.obs.sdf)
                * is_hold_item
            )

        return AgentObservation(
            agent_state=state.agent_state,
            obs_scans=obs_pos,
            life=life,
            is_hold_item=is_hold_item,
            relative_positions=relative_positions,
            intentions=intentions,
            hold_item_info=hold_item_info,
            other_agent_life=other_agent_life,
            item_info=item_info,
            masks=agent_masks,
            item_masks=item_masks,
            item_time=item_time,
            item_starts=item_starts,
            item_goals=item_goals,
            planner_act=planner_act,
        )

    return jax.jit(_observe)


def _build_compute_item_times(env_info):
    num_items = env_info.num_items

    def _inner_compute_item_time(load_item_id: Array, item_time: Array) -> Array:
        """
        Outputs carrying item elapsed time since spawning if the agent is carrying an item
        Outputs 0 if the agent isn't carrying
        To be vmap

        Args:
            load_item_id (Array): ID of the item being carried by the agent.
            item_time (Array): all item elapsed time

        Returns:
            Array: item time
        """
        is_carrying_item = load_item_id < num_items
        item_goal = item_time[load_item_id]
        return item_goal * is_carrying_item

    def _compute_item_time(load_item_id: Array, item_time: Array) -> Array:
        return jax.vmap(_inner_compute_item_time, in_axes=(0, None))(
            load_item_id, item_time
        )

    return jax.jit(_compute_item_time)


def _build_compute_item_goals(env_info):
    num_items = env_info.num_items

    def _inner_compute_item_goals(load_item_id: Array, item_goals: Array) -> Array:
        """
        Outputs the goal of the item the agent is carrying if the agent is carrying an item
        Outputs [0,0] if the agent isn’t carrying
        To be vmap

        Args:
            load_item_id (Array): ID of the item being carried by the agent.
            item_goals (Array): all item goal positions

        Returns:
            Array: item goals
        """
        is_carrying_item = load_item_id < num_items
        item_goal = item_goals[load_item_id] + jnp.array([0, 1])
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


def _build_compute_hold_item_info(env_info: EnvInfo):
    num_agents = env_info.num_agents
    use_load_item_info = env_info.use_hold_item_info

    def _compute_hold_item_info(
        is_hold_item: Array, item_goal: Array, item_time: Array
    ) -> Array:
        """
        compute information of item held by other agents.
        if other agent don't hold any item, then return 0-array.

        Args:
            is_hold_item (Array): whether each agent hold item or not
            item_goal (Array): goal of item held by other agent
            item_time (Array): total elapsed time of item held by other agent

        Returns:
            Array: (is_hold_item, item_goal, item_time). shape: (num_agents, num_agents, 4)

        Note:
            if `env_info.use_hold_item_info` is False, then this funtion return empty array
        """

        if use_load_item_info:
            load_item_info = jnp.concatenate(
                (is_hold_item, item_goal, item_time.reshape(-1, 1)), axis=-1
            )
            load_item_info = jnp.tile(load_item_info.flatten(), num_agents).reshape(
                num_agents, num_agents, -1
            )
        else:
            load_item_info = jnp.zeros((num_agents, num_agents, 0))
        return load_item_info

    return jax.jit(_compute_hold_item_info)
