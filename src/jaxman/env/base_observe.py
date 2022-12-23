"""base observation functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from functools import partial
from typing import Callable, Union

import jax
import jax.numpy as jnp
from chex import Array

from .core import AgentInfo, AgentState, EnvInfo
from .pick_and_delivery.core import State
from .utils import get_scans


def _build_extract_fov(env_info: EnvInfo):
    is_diff_drive = env_info.is_diff_drive
    fov_r = env_info.fov_r
    own_position_map = jnp.zeros((fov_r * 2 + 1, fov_r * 2 + 1)).at[fov_r, fov_r].set(1)

    def rot90_traceable(fov, rot):
        rot = (rot + 2) % 4
        return jax.lax.switch(rot, [partial(jnp.rot90, fov, rot=i) for i in range(4)])

    def _extract_fov(state: AgentState, obs_agent_map: Array) -> Array:
        """
        extract agent fov from obs_agent_map without own position occupancy

        Args:
            state (AgentState): agent state
            obs_agent_map (Array): obstacles and agents map

        Returns:
            Array: agent fov
        """
        x = state.pos[0] + fov_r
        y = state.pos[1] + fov_r
        fov = jax.lax.dynamic_slice(
            obs_agent_map, (x - fov_r, y - fov_r), (fov_r * 2 + 1, fov_r * 2 + 1)
        )
        fov = fov - own_position_map
        if is_diff_drive:
            fov = rot90_traceable(fov, state.rot[0])

        return fov

    return jax.jit(_extract_fov)


def _build_add_agent_pos_to_obstacle_map(env_info) -> Callable:
    fov_r = env_info.fov_r

    def _add_agent_pos_to_obstable_map(pos: Array, obs_map: Array) -> Array:
        batched_obs_pos_map = jax.vmap(
            lambda pos, map: map.at[pos[0] + fov_r, pos[1] + fov_r].set(1.0),
            in_axes=(0, None),
        )(pos, obs_map)
        return jnp.sum(batched_obs_pos_map, axis=0).astype(bool)

    return jax.jit(_add_agent_pos_to_obstable_map)


def batched_apply_rotation(batched_pos: Array, batched_ang: Array) -> Array:
    """apply rotation for batch data

    Args:
        batched_pos (Array): batched agent position
        batched_ang (Array): bathced agent angle

    Returns:
        Array: rotated position
    """

    def apply_rotation(relative_pos: Array, ang: Array) -> Array:
        """apply rotation matrix

        Args:
            relative_pos (Array): relative position of other agents
            ang (Array): agent's own angle

        Returns:
            Array: rotated relative position
        """
        rot_mat = jnp.array(
            [[jnp.cos(ang), jnp.sin(ang)], [-jnp.sin(ang), jnp.cos(ang)]]
        )
        return jnp.dot(relative_pos, rot_mat)

    return jax.vmap(apply_rotation)(batched_pos, batched_ang)


def _build_compute_relative_rot(env_info: EnvInfo):
    def _compute_relative_rot(
        base_state: AgentState, target_state: AgentState
    ) -> Array:
        """compute relative rotation (angle of agent) of all agents

        Args:
            base_state (AgentState): origin state for calculate relative rotation
            target_state (AgentState): target state for calulate relative rotation

        Returns:
            Array: relative rotation
        """
        relative_rot = jax.vmap(lambda target, base: target - base, in_axes=(None, 0))(
            target_state.rot, base_state.rot
        )

        if not env_info.is_discrete:
            return (relative_rot + jnp.pi) % (2 * jnp.pi)
        else:
            return (relative_rot + 2) % 4

    return jax.jit(_compute_relative_rot)


def _build_compute_relative_positions(env_info: EnvInfo):
    is_discrete = env_info.is_discrete
    is_diff_drive = env_info.is_diff_drive

    def _compute_relative_pos(base_state: AgentState, target_pos: Array) -> Array:
        """compute relative position of all agents

        Args:
            base_state (AgentState): origin state for calculate relative position
            target_pos (Array): target position for calulate relative position

        Returns:
            Array: relative position
        """

        relative_pos = jax.vmap(lambda target, base: target - base, in_axes=(None, 0))(
            target_pos, base_state.pos
        )
        if is_diff_drive:
            ang = base_state.rot.flatten() * jnp.pi / 2 - jnp.pi
        else:
            ang = base_state.rot.flatten() - jnp.pi
        if (not is_discrete) or is_diff_drive:
            return batched_apply_rotation(relative_pos, ang)
        else:
            return relative_pos

    return jax.jit(_compute_relative_pos)


def _build_get_other_agent_infos(env_info: EnvInfo):
    _compute_relative_pos = _build_compute_relative_positions(env_info)
    _compute_relative_rot = _build_compute_relative_rot(env_info)

    def _compute_relative_vel(
        base_state: AgentState, target_state: AgentState
    ) -> Array:
        """compute ralative velocity of all agents

        Args:
            base_state (AgentState): origin state for calculate relative velocity
            target_state (AgentState): target state for calulate relative velocity

        Returns:
            Array: relative velocity
        """
        base_rot = (base_state.rot + base_state.ang) % (2 * jnp.pi)
        target_rot = (target_state.rot + target_state.ang) % (2 * jnp.pi)

        base_vel = base_state.vel * jnp.hstack([jnp.sin(base_rot), jnp.cos(base_rot)])
        target_vel = target_state.vel * jnp.hstack(
            [jnp.sin(target_rot), jnp.cos(target_rot)]
        )

        relative_vel = jax.vmap(lambda target, base: target - base, in_axes=(None, 0))(
            target_vel, base_vel
        )
        return relative_vel

    def _get_other_agent_infos(
        base_state: AgentState, target_state: AgentState
    ) -> Array:
        """get all agent's current information

        Args:
            state (AgentState): agent's current state

        Returns:
            Array: all agent current informations
        """
        relative_pos = _compute_relative_pos(base_state, target_state.pos)

        if not env_info.is_discrete:
            relative_rot = _compute_relative_rot(base_state, target_state)
            relative_vel = _compute_relative_vel(base_state, target_state)
            return jnp.concatenate([relative_pos, relative_rot, relative_vel], axis=-1)
        elif env_info.is_diff_drive:
            relative_rot = _compute_relative_rot(base_state, target_state)
            return jnp.concatenate((relative_pos, relative_rot), axis=-1)
        else:
            return relative_pos

    return jax.jit(_get_other_agent_infos)


def _build_compute_neighbor_mask(
    env_info: EnvInfo,
):
    num_agents = env_info.num_agents
    r = env_info.comm_r
    base_mask = ~jnp.eye(num_agents, dtype=bool)

    def _compute_neighbor_mask(relative_pos: Array, not_finished_agent: Array) -> Array:
        """
        compute mask for obtaining only neighboring agent communications.
        neighbor is defined by distance between each agents.

        Args:
            relative_pos (Array): relative position between all agents
            not_finished_agent (Array): array on whether each agent completed the their episode.

        Returns:
            Array: mask specifying neighboring agents
        """
        agent_dist = jnp.sqrt(jnp.sum(relative_pos**2, axis=-1))
        neighbor_mask = agent_dist < r
        neighbor_mask = neighbor_mask * base_mask
        neighbor_done_mask = jax.vmap(lambda a, b: a * b, in_axes=(0, None))(
            neighbor_mask, not_finished_agent
        )
        return neighbor_done_mask

    return jax.jit(_compute_neighbor_mask)


### obstacle ditection ###
def _build_get_obs_pos(env_info: EnvInfo, agent_info: AgentInfo):
    def _build_get_discrete_obs_pos(env_info: EnvInfo):
        fov_r = env_info.fov_r
        num_agents = env_info.num_agents
        _extract_fov = _build_extract_fov(env_info)
        _add_agent_pos_to_obstacle_map = _build_add_agent_pos_to_obstacle_map(env_info)

        def _get_obs_and_agent_pos(
            state: Union[AgentState, State], obs_map: Array
        ) -> Array:
            """
            get flatten neighboring obstacles and agent position

            Args:
                state (AgentState): agent's current state
                obs_map (Array): obstacle map. obs_map is added One padding

            Returns:
                Array: flatten obs and agent position
            """
            obs_map = jnp.pad(obs_map, fov_r, mode="constant", constant_values=0)
            if env_info.env_name == "navigation":
                pos = state.pos
            else:
                pos = jnp.vstack((state.agent_state.pos, state.item_pos))
                state = state.agent_state
            obs_agent_map = _add_agent_pos_to_obstacle_map(pos, obs_map)
            fov = jax.vmap(_extract_fov, in_axes=(0, None))(state, obs_agent_map)
            flatten_fov = fov.reshape(num_agents, -1)
            return flatten_fov

        return jax.jit(_get_obs_and_agent_pos)

    def _build_get_continous_obs_pos(env_info: EnvInfo, agent_info: AgentInfo):
        rads = jnp.array(agent_info.rads)
        num_scans = env_info.num_scans
        scan_range = env_info.scan_range
        agent_index = jnp.arange(env_info.num_agents)

        def _get_obs_and_agent_pos(
            state: AgentState, all_state: AgentState, edges: Array, agent_index: Array
        ) -> Array:
            """get obstacle and other agent position as lidar scan for agent_index th agent.

            Args:
                state (AgentState): agent state
                all_state (AgentState): all agent state
                edges (Array): obstacles edges
                agent_index (Array): agent index

            Returns:
                Array: lidar scan of neighboring obstacles and agent
            """
            rads_wo_self = rads.at[agent_index].set(0)
            top_left = all_state.pos - rads_wo_self
            bottom_right = all_state.pos + rads_wo_self
            top_right = jnp.hstack((top_left[:, 0:1], bottom_right[:, 1:2]))
            bottom_left = jnp.hstack((bottom_right[:, 0:1], top_left[:, 1:2]))
            a = jnp.stack((top_left, bottom_left), axis=1)
            b = jnp.stack((bottom_left, bottom_right), axis=1)
            c = jnp.stack((bottom_right, top_right), axis=1)
            d = jnp.stack((top_right, top_left), axis=1)

            edges = jnp.concatenate((env_info.edges, a, b, c, d))
            scans = get_scans(state.pos, state.rot, edges, num_scans, scan_range)
            return scans

        def _get_all_agent_obs_agent_pos(state: AgentState, edges: Array) -> Array:
            """get obstacle and other agent position"""
            scans = jax.vmap(_get_obs_and_agent_pos, in_axes=(0, None, None, 0))(
                state, state, edges, agent_index
            )
            return scans

        return _get_all_agent_obs_agent_pos

    if env_info.is_discrete:
        _get_obs_pos = _build_get_discrete_obs_pos(env_info)
    else:
        _get_obs_pos = _build_get_continous_obs_pos(env_info, agent_info)
    return _get_obs_pos
