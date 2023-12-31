"""kinematic dynamics

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo

Credits: Code in this file is based on https://github.com/omron-sinicx/jaxmapp
"""

from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array

from .core import AgentInfo, AgentState, EnvInfo
from .obstacle import ObstacleMap
from .utils import xy_to_ij


def _build_compute_next_state(env_info: EnvInfo):
    discrete_action_list = jnp.array(
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int
    )

    def _compute_continuous_next_state(
        state: AgentState,
        actions: Array,
        agent_info: AgentInfo,
    ) -> AgentState:
        """
        compute continuous agent next state.
        Supposed to be used with 'jax.vmap'

        Args:
            state (AgentState): agent current state
            actions (Array): agent action, value_range: -1<actions<1
            agent_info (AgentInfo): agent kinematics information

        Returns:
            AgentState: next agent state
        """
        if env_info.use_acc:
            vel = jnp.clip(
                state.vel + actions[0] * agent_info.max_accs,
                a_min=0,
                a_max=agent_info.max_vels,
            )
            ang = jnp.clip(
                state.ang + actions[1] * agent_info.max_ang_accs,
                a_min=-agent_info.max_ang_vels,
                a_max=agent_info.max_ang_vels,
            )
        else:
            # Convert -1<action<1 into 0<action<1
            normed_action = (actions[0] + 1) / 2
            vel = normed_action * agent_info.max_vels
            ang = actions[1] * agent_info.max_ang_vels
        next_rot = (state.rot + ang) % (2 * jnp.pi)
        next_pos = state.pos + vel * jnp.hstack([jnp.sin(next_rot), jnp.cos(next_rot)])
        next_state = AgentState(pos=next_pos, rot=next_rot, vel=vel, ang=ang)
        next_state.is_valid()
        return next_state

    def _compute_grid_next_state(
        state: AgentState,
        action: Array,
        _,
    ) -> AgentState:
        """
        compute grid agent next state
        Supposed to be used with 'jax.vmap'

        Args:
            state (AgentState): agent current state
            action (Array): agent action
            _ : dummy variable to align number of variables

        Returns:
            AgentState: agent next state
        """

        actions = discrete_action_list[action]
        next_pos = state.pos + actions.astype(int)
        next_state = AgentState(
            pos=next_pos, rot=state.rot, vel=state.vel, ang=state.ang
        )
        next_state.is_valid()
        return next_state

    if not env_info.is_discrete:
        return _compute_continuous_next_state
    else:
        return _compute_grid_next_state


### collide ###
# agent collision ditection
def _build_check_collision_with_agents(
    env_info: EnvInfo, agent_info: AgentInfo, is_discrete: bool
) -> Callable:
    if is_discrete:
        _check_agent_collide = _build_check_discrete_collision_with_agent(env_info)
    else:
        _check_agent_collide = _build_check_continous_collision_with_agent(
            env_info, agent_info
        )
    return _check_agent_collide


def _build_check_discrete_collision_with_agent(env_info: EnvInfo):

    agent_index = jnp.arange(env_info.num_agents)

    def _check_collision_at_same_pos(
        all_pos: Array, query_pos: Array, self_id: int
    ) -> Array:
        """
        Check collision to all agents but that specified by `self_id`
        this function ditect agent collision when multiple agent move into the same position

        Args:
            all_pos (Array): all agent's positions
            query_pos (Array): query position
            self_id (int): index of agent to be omitted from distance computation (i.e., self)

        Returns:
            Array: agent collide
        """
        all_state_pos = all_pos.at[self_id].set(jnp.inf)
        agent_collide = jnp.any(jnp.all(jnp.equal(all_state_pos, query_pos), axis=-1))
        return agent_collide

    def _check_collision_by_crossing(all_pos: Array, all_pos_next: Array) -> Array:
        """
        Check collision to all agents
        this function ditect agent collision when multiplt agent crossing over the same position

        Args:
            all_pos (Array): all agent position
            all_pos_next (Array): all agent next position

        Returns:
            Array: collided or not
        """

        def _at_old_pos(all_pos: Array, pos_next: Array, self_id: int):
            all_pos = all_pos.at[self_id].set(jnp.inf)
            at_old_pos = jnp.all(jnp.equal(all_pos, pos_next), axis=-1)
            return at_old_pos

        at_old_pos = jax.vmap(_at_old_pos, in_axes=(None, 0, 0))(
            all_pos, all_pos_next, agent_index
        )
        crossing_collide = jnp.any(jnp.logical_and(at_old_pos, at_old_pos.T), axis=-1)
        return crossing_collide

    def _check_collision_with_agent(all_pos: Array, all_pos_next: Array):
        """
        Check collision to all agents
        this function ditect agent collision when multiplt agent move to the same position

        Args:
            all_pos (Array): all agent position
            all_pos_next (Array): all agent next position

        Returns:
            Array: collided or not
        """
        same_pos_collide = jax.vmap(_check_collision_at_same_pos, in_axes=(None, 0, 0))(
            all_pos_next, all_pos_next, agent_index
        )
        crossing_collide = _check_collision_by_crossing(all_pos, all_pos_next)
        agent_collide = same_pos_collide | crossing_collide

        return agent_collide

    return jax.jit(_check_collision_with_agent)


def _get_agent_dist(
    all_pos: Array, query_pos: Array, agent_info: AgentInfo, self_id: int
) -> Array:
    """
    Get distance to all agents but that specified by `self_id`

    Args:
        all_pos (Array): all agent's positions
        query_pos (Array): query position
        agent_info (AgentInfo): agent base information
        self_id (int): index of agent to be omitted from distance computation (i.e., self)

    Returns:
        Array: distance matrix
    """
    all_state_pos = all_pos.at[self_id].set(jnp.inf)
    agent_dist = (
        jnp.linalg.norm(all_state_pos - query_pos, axis=1) - agent_info.rads.flatten()
    )
    return agent_dist


def _build_check_continous_collision_with_agent(
    env_info: EnvInfo, agent_info: AgentInfo
):

    agent_index = jnp.arange(env_info.num_agents)

    def _check_collision_wiht_agents(_: Array, all_pos_next: Array) -> Array:
        """
        check collision to all agent.
        this function ditect agent collision when distance to other agents is smaller than agent radius

        Args:
            _ (Array): dummy variable.
            all_pos_next (Array): all agent next position

        Returns:
            Array: collided or not
        """
        agent_dist = jax.vmap(_get_agent_dist, in_axes=(None, 0, None, 0))(
            all_pos_next, all_pos_next, agent_info, agent_index
        )
        agent_collide = jnp.any(agent_dist < agent_info.rads, axis=-1)
        return agent_collide

    return jax.jit(_check_collision_wiht_agents)


# obstacle collide ditection
def _build_check_collision_with_obs(agent_info: AgentInfo, is_discrete: bool):
    if is_discrete:
        _check_collision_with_obs = _build_check_discrete_collision_with_obs()
    else:
        _check_collision_with_obs = _build_check_continuous_collision_with_obs(
            agent_info
        )
    return _check_collision_with_obs


def _build_check_discrete_collision_with_obs():
    def _check_collision_with_obs(pos: Array, obs: ObstacleMap) -> Array:
        """
        get obstacle collide base on obstable map used for discrete agent

        Args:
            pos (Array): An agent's position
            obs_map (Array): obstacle map

        Returns:
            Array: collided or not
        """
        obs_collide = jnp.array(obs.occupancy[pos[:, 0], pos[:, 1]], dtype=bool)
        return obs_collide

    return jax.jit(_check_collision_with_obs)


def _get_obstacle_dist(pos: Array, sdf_map: Array) -> float:
    """
    Get distance to the closest obstacles based on precomputed SDF used for continuous agent

    Args:
        pos (Array): An agent's position

    Returns:
        float: distance from the query state to the closest obstacle
    """
    # chex.assert_shape(pos, (2,))
    pos_ij = xy_to_ij(pos, sdf_map.shape[0])
    sdf_map = jnp.array(sdf_map)

    sdf_dist = sdf_map[pos_ij[0], pos_ij[1]]
    return sdf_dist


def _build_check_continuous_collision_with_obs(agent_info: AgentInfo):
    def _check_collision_with_obs(pos: Array, obs: ObstacleMap) -> Array:
        """
        get obstacle collision based on precomputed SDF

        Args:
            pos (Array): An agent's position
            sdf_map (Array): precoputed SDF map

        Returns:
            Array: collided or not
        """
        dist_to_obs = jax.vmap(_get_obstacle_dist, in_axes=(0, None))(pos, obs.sdf)
        obs_collided = dist_to_obs < agent_info.rads.reshape(-1)
        return obs_collided

    return jax.jit(_check_collision_with_obs)


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


def _build_get_relative_positions(env_info: EnvInfo):
    is_discrete = env_info.is_discrete

    def _compute_relative_pos(
        base_state: AgentState, target_state: AgentState
    ) -> Array:
        """compute relative position of all agents

        Args:
            base_state (AgentState): origin state for calculate relative position
            target_state (AgentState): target state for calulate relative position

        Returns:
            Array: relative position
        """

        relative_pos = jax.vmap(lambda target, base: target - base, in_axes=(None, 0))(
            target_state.pos, base_state.pos
        )
        ang = base_state.rot.flatten() - jnp.pi
        if not is_discrete:
            return batched_apply_rotation(relative_pos, ang)
        else:
            return relative_pos

    return jax.jit(_compute_relative_pos)


def _build_compute_neighbor_agent_mask(env_info: EnvInfo):
    num_agents = env_info.num_agents
    r = env_info.comm_r
    base_mask = ~jnp.eye(num_agents, dtype=bool)

    def _compute_neighbor_agent_mask(
        relative_pos: Array, not_finished_agent: Array
    ) -> Array:
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

    return jax.jit(_compute_neighbor_agent_mask)
