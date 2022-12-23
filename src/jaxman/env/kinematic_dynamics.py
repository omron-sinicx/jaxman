"""kinematic dynamics

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from chex import Array

from .core import AgentInfo, AgentState, EnvInfo
from .obstacle import ObstacleMap
from .utils import xy_to_ij


def _build_compute_next_state(is_discrete: bool, is_diff_drive: bool):
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
            actions (Array): agent action
            agent_info (AgentInfo): agent kinematics information

        Returns:
            AgentState: next agent state
        """
        vel = jnp.clip(
            state.vel + actions[0] * agent_info.max_accs,
            a_min=agent_info.min_vels,
            a_max=agent_info.max_vels,
        )
        ang = jnp.clip(
            state.ang + actions[1] * agent_info.max_ang_accs,
            a_min=agent_info.min_ang_vels,
            a_max=agent_info.max_ang_vels,
        )
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

    def _compute_diff_drive_next_state(
        state: AgentState,
        action: Array,
        _: AgentInfo,
    ) -> AgentState:
        """
        compute diff drive agent next state

        Args:
            state (AgentState): agent current state
            action (Array): agent action
            _ : dummy variable to align number of variables

        Returns:
            AgentState: agent next state
        """

        def compute_actions(action: Array, rot: Array) -> Tuple[Array, Array]:
            """compute pos and rot action in diff drive agent

            Args:
                action (Array): agent raw action
                rot (Array): agent current rotation

            Returns:
                Array: [pos_action, rot_action]
            """
            # Stay
            def action_0(rot):
                return jnp.array([0, 0], dtype=int), jnp.array([0], dtype=int)

            # Go
            def action_1(rot):
                pos_action = move_with_rot(rot, 1)
                rot_action = jnp.array([0], dtype=int)
                return pos_action, rot_action

            # Back
            def action_2(rot):
                pos_action = move_with_rot(rot, -1)
                rot_action = jnp.array([0], dtype=int)
                return pos_action, rot_action

            # Turn Left
            def action_3(rot):
                return jnp.array([0, 0], dtype=int), jnp.array([-1], dtype=int)

            # Turn Right
            def action_4(rot):
                return jnp.array([0, 0], dtype=int), jnp.array([1], dtype=int)

            def move_with_rot(rot, way):
                def move_0():
                    return jnp.array([0, 1], dtype=int)

                def move_1():
                    return jnp.array([1, 0], dtype=int)

                def move_2():
                    return jnp.array([0, -1], dtype=int)

                def move_3():
                    return jnp.array([-1, 0], dtype=int)

                direction = jax.lax.switch(rot, [move_0, move_1, move_2, move_3])
                return (direction * way).astype(int)

            return jax.lax.switch(
                action, [action_0, action_1, action_2, action_3, action_4], rot
            )

        pos_action, rot_action = compute_actions(action, state.rot[0])
        next_pos = state.pos + pos_action
        next_rot = (state.rot + rot_action) % 4
        vel = jnp.array([0], dtype=int)
        ang = jnp.array([0], dtype=int)
        next_state = AgentState(pos=next_pos, rot=next_rot, vel=vel, ang=ang)
        next_state.is_valid()
        return next_state

    if not is_discrete:
        return _compute_continuous_next_state
    elif is_diff_drive:
        return _compute_diff_drive_next_state
    else:
        return _compute_grid_next_state


### collide ###
# agent collide ditection
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
    is_diff_drive = env_info.is_diff_drive

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
        if is_diff_drive:
            ang = base_state.rot.flatten() * jnp.pi / 2 - jnp.pi
        else:
            ang = base_state.rot.flatten() - jnp.pi
        if (not is_discrete) or is_diff_drive:
            return batched_apply_rotation(relative_pos, ang)
        else:
            return relative_pos

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

        if not is_discrete:
            return (relative_rot + jnp.pi) % (2 * jnp.pi)
        else:
            return (relative_rot + 2) % 4

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

    def _get_relative_position(
        base_state: AgentState, target_state: AgentState
    ) -> Array:
        """get all agent's current information

        Args:
            state (AgentState): agent's current state

        Returns:
            Array: all agent current informations
        """
        relative_pos = _compute_relative_pos(base_state, target_state)

        if not is_discrete:
            relative_rot = _compute_relative_rot(base_state, target_state)
            relative_vel = _compute_relative_vel(base_state, target_state)
            return jnp.concatenate([relative_pos, relative_rot, relative_vel], axis=-1)
        elif is_diff_drive:
            relative_rot = _compute_relative_rot(base_state, target_state)
            return jnp.concatenate((relative_pos, relative_rot), axis=-1)
        else:
            return relative_pos

    return jax.jit(_get_relative_position)


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
