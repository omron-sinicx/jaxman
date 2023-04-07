"""Environment dynamics functions for Pick and Delivery Env

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, assert_shape

from ..core import AgentInfo, AgentState, EnvInfo
from ..kinematic_dynamics import (
    _build_check_collision_with_agents,
    _build_check_collision_with_obs,
    _build_compute_next_state,
    _get_obstacle_dist,
)
from ..obstacle import ObstacleMap
from .core import State, TaskInfo, TrialInfo

INF = 10000


def _build_inner_step(env_info: EnvInfo, agent_info: AgentInfo) -> Callable:
    """build step function

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information

    Returns:
        Callable: jit-compiled observe function
    """
    is_discrete = env_info.is_discrete
    is_crashable = env_info.is_crashable
    _compute_next_state = _build_compute_next_env_state(env_info, agent_info)
    _check_collision_wiht_agents = _build_check_collision_with_agents(
        env_info, agent_info, is_discrete
    )
    _check_collision_with_obs = _build_check_collision_with_obs(agent_info, is_discrete)
    _check_collision_with_items = _build_check_collision_with_items(
        env_info, agent_info
    )
    _compute_is_item_solved = _build_compute_is_solved(env_info, agent_info)
    _check_is_item_collided = _build_check_is_item_collided(env_info, agent_info)
    _compute_rew_done_info = _build_compute_rew_done_info(env_info)
    _respawn_items = _build_respawn_items(env_info)

    def _inner_step(
        key: PRNGKey,
        state: State,
        actions: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
    ) -> Tuple[PRNGKey, State, Array, Array, TaskInfo, TrialInfo]:
        """inner step function. inner step function consisting of the following steps:
        1) compute next state by agent's movement action (don't consider load/unload action)
        2) update agent and item state by agent's load/unload action
        3) check if any agent collides with obstacles/other agents
        4) check if item is delivered at its goal
        5) create return vals

        Args:
            key (PRNGKey): random variable key
            state (State): current agent and item state
            actions (Array): actions
            task_info (TaskInfo): task information
            trial_info (TrialInfo): trial information

        Returns:
            Tuple[State, Array, Array, TrialInfo]: next_state, rewards, dones, new trial information
        """

        # update state
        movable_agents = jnp.logical_not(trial_info.agent_collided)
        masked_actions = jax.vmap(lambda a, b: a * b)(actions, movable_agents)
        possible_next_state, new_item_starts = _compute_next_state(
            state, masked_actions, task_info
        )

        # update agent collision
        collided_with_obs = _check_collision_with_obs(
            possible_next_state.agent_state.pos, task_info.obs
        )
        collide_with_agent = _check_collision_wiht_agents(
            state.agent_state.pos, possible_next_state.agent_state.pos
        )
        collide_with_item = _check_collision_with_items(
            possible_next_state.agent_state.pos, possible_next_state.item_pos
        )
        agent_collided = jnp.logical_or(
            jnp.logical_or(collided_with_obs, collide_with_agent), collide_with_item
        )

        # update item trial_info
        item_solved = _compute_is_item_solved(
            possible_next_state.item_pos, task_info.item_goals
        )
        item_collided = _check_is_item_collided(possible_next_state, agent_collided)

        rews, done, new_trial_info = _compute_rew_done_info(
            state,
            possible_next_state,
            agent_collided * is_crashable,
            item_collided * is_crashable,
            item_solved,
            trial_info,
            task_info.item_starts,
            new_item_starts,
            task_info.item_goals,
        )

        # if agent finish own episode, agent speed is set to 0
        not_finished_agents = jnp.expand_dims(jnp.logical_not(agent_collided), axis=-1)
        vel = possible_next_state.agent_state.vel * not_finished_agents
        ang = possible_next_state.agent_state.ang * not_finished_agents
        new_agent_state = possible_next_state.agent_state._replace(vel=vel, ang=ang)

        # if item finish episode, set item position to INF
        done_item = jnp.logical_or(item_collided * is_crashable, item_solved)

        item_pos = jax.vmap(lambda pos, done: pos + done * INF)(
            possible_next_state.item_pos, done_item.astype(bool)
        )

        item_time = possible_next_state.item_time * jnp.logical_not(done_item)

        next_state = possible_next_state._replace(
            agent_state=new_agent_state, item_pos=item_pos, item_time=item_time
        )

        # respawn items
        task_info = task_info._replace(item_starts=new_item_starts)
        if env_info.is_respawn:
            key, next_state, task_info = _respawn_items(
                key, next_state, task_info, done_item.astype(bool)
            )

        return key, next_state, rews, done, task_info, new_trial_info

    return jax.jit(_inner_step)


def _build_compute_next_env_state(env_info: EnvInfo, agent_info: AgentInfo) -> Callable:
    """build function to compute next agent and item state

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information

    Returns:
        Callable: jit-compiled function to compute next agent and item state
    """
    _compute_agent_next_state = _build_compute_next_agent_state(env_info, agent_info)
    _compute_next_item_state = _build_compute_next_item_state(env_info, agent_info)

    def _compute_next_env_state(
        state: State, actions: Array, task_info: TaskInfo
    ) -> State:
        next_agent_state = _compute_agent_next_state(
            state, actions, agent_info, task_info
        )
        next_load_item_id, next_item_pos, item_starts = _compute_next_item_state(
            next_agent_state, actions, state.load_item_id, state.item_pos, task_info
        )
        is_load_item = next_load_item_id < env_info.num_items
        life = state.life - is_load_item + jnp.logical_not(is_load_item)
        life = jnp.clip(life, a_min=-1, a_max=5)
        item_time = state.item_time + 1
        return (
            State(next_agent_state, next_load_item_id, life, next_item_pos, item_time),
            item_starts,
        )

    return jax.jit(_compute_next_env_state)


def _build_compute_next_agent_state(
    env_info: EnvInfo, agent_info: AgentInfo
) -> Callable:
    """
    build jit-compiled function to compute next agent state

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information

    Returns:
        Callable: jit-compiled function
    """
    is_discrete = env_info.is_discrete
    num_agents = env_info.num_agents
    _compute_kinematic_next_state = _build_compute_next_state(
        env_info.is_discrete, env_info.is_diff_drive
    )
    _check_collide_with_agent_item = _build_check_collide_with_agent_item(
        env_info, agent_info
    )
    _check_collide_with_obs = _build_check_collide_with_obs(agent_info)

    def _apply_load_or_unload(
        current_state: State, next_possible_state: AgentState, action: Array
    ) -> AgentState:
        """
        update agent state by considering load/unload action
        if agent choice load/unload action agent try to load/unload item and stay still the current position

        Args:
            current_state (State): agent current state
            next_possible_state (AgentState): agent next state
            action (Array): action

        Returns:
            AgentState: updated agent next state
        """
        if is_discrete:
            can_move = action < 5
            arrayed_next_state = (
                can_move * next_possible_state.cat() + ~can_move * current_state.cat()
            )
            next_state = AgentState.from_array(arrayed_next_state.astype(int))
        else:
            can_move = action[-1] <= 0
            # if agent choose to load/unload action, then agent stop at the current position and vel and ang is set to 0
            next_state = AgentState(
                pos=can_move * next_possible_state.pos + ~can_move * current_state.pos,
                rot=can_move * next_possible_state.rot + ~can_move * current_state.rot,
                vel=can_move * next_possible_state.vel,
                ang=can_move * next_possible_state.ang,
            )
        return next_state

    def _compute_agent_next_state(
        state: State, actions: Array, agent_info: AgentInfo, task_info: TaskInfo
    ) -> AgentState:
        if is_discrete:
            possible_next_state = jax.vmap(_compute_kinematic_next_state)(
                state.agent_state, actions, agent_info
            )
        else:
            possible_next_state = jax.vmap(_compute_kinematic_next_state)(
                state.agent_state, actions[:-1], agent_info
            )
        next_state = jax.vmap(_apply_load_or_unload)(
            state.agent_state, possible_next_state, actions
        )
        return next_state

    def _compute_discrete_non_crashable_next_state(
        state: State, action: Array, agent_info: AgentInfo, task_info: TaskInfo
    ) -> AgentState:
        class Carry(NamedTuple):
            state: Array
            agent_obs_item_map: Array
            collided: Array

        def _inner_step(i: int, carry: Carry):
            next_state = _compute_kinematic_next_state(
                AgentState.from_array(carry.state[i]), action[i], agent_info.at(i)
            )

            next_pos = next_state.pos
            is_move_action = action[i] < 5
            is_collide = jnp.array(
                carry.agent_obs_item_map[next_pos[0], next_pos[1]], dtype=bool
            )
            can_move = jnp.logical_and(is_move_action, jnp.logical_not(is_collide))

            next_state = can_move * next_state.cat() + ~can_move * carry.state[i]

            state = carry.state.at[i].set(next_state)
            agent_obs_item_map = carry.agent_obs_item_map.at[
                next_state[0], next_state[1]
            ].set(1)
            collided = carry.collided.at[i].set(is_collide)
            return Carry(state, agent_obs_item_map, collided)

        agent_obs_item_map = jnp.sum(
            jax.vmap(
                lambda pos, map: map.at[pos[0], pos[1]].set(1.0),
                in_axes=(0, None),
            )(state.item_pos, task_info.obs.occupancy),
            axis=0,
        )
        is_collided = jnp.zeros((env_info.num_agents,))
        carry = Carry(state.agent_state.cat(), agent_obs_item_map, is_collided)
        carry = jax.lax.fori_loop(0, env_info.num_agents, _inner_step, carry)
        return AgentState.from_array(carry.state)

    def _compute_continuous_non_crashable_next_state(
        state: State, action: Array, agent_info: AgentInfo, task_info: TaskInfo
    ) -> AgentState:
        class Carry(NamedTuple):
            state: Array
            agent_pos: Array
            collided: Array

        def _inner_step(i: int, carry: Carry) -> Carry:
            next_state = _compute_kinematic_next_state(
                AgentState.from_array(carry.state[i]), action[i, :-1], agent_info.at(i)
            )

            next_pos = next_state.pos
            is_move_action = action[i][-1] <= 0
            # check collision with agents or items
            all_pos = jnp.concatenate((carry.agent_pos, item_pos))
            agent_item_collided = _check_collide_with_agent_item(next_pos, all_pos)
            # check collision with obstacles
            obs_collided = _check_collide_with_obs(next_pos, task_info.obs.sdf)
            is_collide = jnp.logical_or(agent_item_collided, obs_collided)
            can_move = jnp.logical_and(is_move_action, jnp.logical_not(is_collide))
            # if agent take un movable action, then agent's vel and ang is set to 0
            possible_state = carry.state[i].at[-1].set(0).at[-2].set(0)
            next_state = can_move * next_state.cat() + ~can_move * possible_state

            state = carry.state.at[i].set(next_state)
            agent_pos = carry.agent_pos.at[i].set(next_state[:2])
            collided = carry.collided.at[i].set(is_collide)
            return Carry(state, agent_pos, collided)

        item_pos = state.item_pos
        agent_pos = jnp.ones((num_agents, 2), dtype=float) * jnp.inf
        is_collided = jnp.zeros((num_agents,))
        carry = Carry(state.agent_state.cat(), agent_pos, is_collided)
        carry = jax.lax.fori_loop(0, env_info.num_agents, _inner_step, carry)
        return AgentState.from_array(carry.state)

    if env_info.is_crashable:
        return jax.jit(_compute_agent_next_state)
    else:
        if is_discrete:
            return jax.jit(_compute_discrete_non_crashable_next_state)
        else:
            return jax.jit(_compute_continuous_non_crashable_next_state)


def _build_compute_next_item_state(
    env_info: EnvInfo, agent_info: AgentInfo
) -> Callable:
    """build jit-compiled functino to compute item next state

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematic information

    Returns:
        Callable: jit-compiled function
    """
    num_items = env_info.num_items
    dummy_index = num_items
    num_agents = env_info.num_agents
    _check_collide_with_agent_item = _build_check_collide_with_agent_item(
        env_info, agent_info
    )
    _check_collide_with_obs = _build_check_collide_with_obs(agent_info)
    rads = jnp.array(agent_info.rads)

    class Carry(NamedTuple):
        state: AgentState
        actions: Array
        load_item_id: int
        item_pos: Array
        item_starts: Array
        obs: ObstacleMap

    def _move_or_load_unload_items(i: int, carry: Carry):
        """compute item next state to be jax.lax.for_i_loop

        Args:
            i (int): for_loop agent index
            carry (Carry): information carrier

        Step:
            1) check whether the agent is trying to move
            2.1) if agent is trying to move
                item state remains the same.
            2.2) if agent is trying to load or unload items
                2.2.1) if the agent is currently carrying item
                    if there is a place to unload item, unload item.
                2.2.2) if the agent isn't currently carrying item
                    If there is an item in the neighborhood, load the item
        """

        def move_agent(i: int, carry: Carry):
            """move agent, item state remain the same

            Args:
                carry (Carry): info carry

            Attention:
                The location of the item being carried by the agent is set to INF.
                Reason for this is to prevent other agents from recognizing items that are in the process of being carried.
                Thus, if a particular agent chooses to move, the state of the item does not change.
            """

            return carry

        def load_or_unload(i: int, carry: Carry):
            """
            load or unload item.
            if agent is carrying item, then agent try to unload item, and vice versa

            Args:
                i (int): for_loop agent index
                carry (Carry): information carrier
            """

            def unload_items(i: int, carry: Carry):
                """
                try to unload item
                The agent unloads the item if there is nothing at the destination where the item is to be unloaded;
                otherwise, the item remains loaded.
                """
                if env_info.is_discrete:
                    possible_item_pos = carry.state.pos[i] + jnp.array(
                        [0, -1], dtype=int
                    )
                    # Check whether there are any obstacles at the location to unload
                    obstacle_collide = jnp.array(
                        carry.obs.occupancy[possible_item_pos[0], possible_item_pos[1]],
                        dtype=bool,
                    )
                    # Check whether there are any agent or items at the location to unload
                    agent_item_pos = jnp.concatenate(
                        (carry.state.pos, carry.item_pos), axis=0
                    )
                    agent_item_collide = jnp.any(
                        jnp.all(jnp.equal(agent_item_pos, possible_item_pos), axis=-1)
                    )

                    can_unload = jnp.logical_not(
                        jnp.logical_or(obstacle_collide, agent_item_collide)
                    )

                    # if agent can unload items, item_position is updated, else item position is set to INF (agent continue to load item)
                    next_item_pos = carry.item_pos.at[carry.load_item_id[i]].set(
                        can_unload * possible_item_pos
                        + ~can_unload * jnp.array([INF, INF])
                    )
                    # if agent can unload items, load_item_id for i th agent is set to dummy_index(dot't carry any item)
                    next_load_item_id = carry.load_item_id.at[i].set(
                        can_unload * dummy_index + ~can_unload * carry.load_item_id[i]
                    )

                    return carry._replace(
                        item_pos=next_item_pos, load_item_id=next_load_item_id
                    )
                else:
                    # unload item in the direction the agent is facing
                    possible_item_pos = carry.state.pos[i] + 2.5 * rads[i] * jnp.hstack(
                        [jnp.sin(carry.state.rot[i]), jnp.cos(carry.state.rot[i])]
                    )
                    # Check whether there are any obstacles at the location to unload
                    obstacle_collide = _check_collide_with_obs(
                        possible_item_pos, carry.obs.sdf
                    )
                    # Check whether there are any agents or items at the location to unload
                    all_pos = jnp.concatenate((carry.state.pos, carry.item_pos))
                    agent_item_collide = _check_collide_with_agent_item(
                        possible_item_pos, all_pos
                    )

                    can_unload = jnp.logical_not(
                        jnp.logical_or(obstacle_collide, agent_item_collide)
                    )

                    # if agent can unload items, item_position is updated, else item position is set to INF (agent continue to load item)
                    next_item_pos = carry.item_pos.at[carry.load_item_id[i]].set(
                        can_unload * possible_item_pos
                        + ~can_unload * jnp.array([INF, INF])
                    )
                    # if agent can unload items, load_item_id for i th agent is set to dummy_index(dot't carry any item)
                    next_load_item_id = carry.load_item_id.at[i].set(
                        can_unload * dummy_index + ~can_unload * carry.load_item_id[i]
                    )

                    # if agent can unload item, item_starts is updated to the next state
                    item_starts = carry.item_starts.at[carry.load_item_id[i]].set(
                        can_unload * next_item_pos[carry.load_item_id[i]]
                        + ~can_unload * carry.item_starts[carry.load_item_id[i]]
                    )

                    return carry._replace(
                        item_pos=next_item_pos,
                        load_item_id=next_load_item_id,
                        item_starts=item_starts,
                    )

            def load_items(i: int, carry: Carry):
                """
                load item if there is an item in neighbor
                and remain unloaded if there is no item in neighbor
                """
                dist_to_item = jnp.sqrt(
                    jnp.sum((carry.item_pos - carry.state.pos[i]) ** 2, axis=1)
                )
                nearest_item_index = jnp.argmin(dist_to_item)
                nearest_item_dist = dist_to_item[nearest_item_index]
                if env_info.is_discrete:
                    can_pickup = nearest_item_dist <= 1
                else:
                    can_pickup = nearest_item_dist <= 2.5 * rads[i][0]
                    assert_shape(can_pickup, ())
                next_load_item_id = carry.load_item_id.at[i].set(
                    can_pickup * nearest_item_index + ~can_pickup * dummy_index
                )
                next_item_pos = carry.item_pos.at[nearest_item_index].set(
                    can_pickup * jnp.array([INF, INF])
                    + ~can_pickup * carry.item_pos[nearest_item_index]
                )

                return carry._replace(
                    item_pos=next_item_pos, load_item_id=next_load_item_id
                )

            return jax.lax.cond(
                carry.load_item_id[i] < num_items, unload_items, load_items, i, carry
            )

        if env_info.is_discrete:
            is_move = carry.actions[i] < 5
        else:
            is_move = carry.actions[i, -1] <= 0
        return jax.lax.cond(is_move, move_agent, load_or_unload, i, carry)

    def _compute_next_item_state(
        state: AgentState,
        actions: Array,
        load_item_id: Array,
        item_pos: Array,
        task_info: TaskInfo,
    ) -> Tuple[Array, Array]:
        carry = Carry(
            state, actions, load_item_id, item_pos, task_info.item_starts, task_info.obs
        )
        carry = jax.lax.fori_loop(0, num_agents, _move_or_load_unload_items, carry)
        return carry.load_item_id, carry.item_pos, carry.item_starts

    return jax.jit(_compute_next_item_state)


def _build_check_collision_with_items(env_info: EnvInfo, agent_info: AgentInfo):
    """build jit compiled collision check function

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information
    """

    def _check_discrete_collision_with_items(pos: Array, item_pos: Array):
        collide = jax.vmap(
            lambda pos, item_pos: jnp.any(jnp.all(jnp.equal(item_pos, pos), axis=-1)),
            in_axes=(0, None),
        )(pos, item_pos)
        return collide

    def _check_continuous_collision_with_items(pos: Array, item_pos: Array):
        def _inner_check(pos: Array, item_pos: Array):
            dist_to_item = (
                jnp.linalg.norm(item_pos - pos, axis=1) - env_info.item_rads.flatten()
            )
            assert_shape(dist_to_item, (env_info.num_items,))
            return jnp.any(dist_to_item < env_info.item_rads)

        collide = jax.vmap(_inner_check, in_axes=(0, None))(pos, item_pos)
        return collide

    if env_info.is_discrete:
        return _check_discrete_collision_with_items
    else:
        return _check_continuous_collision_with_items


def _build_check_is_item_collided(env_info: EnvInfo, agent_info: AgentInfo):
    """check whether item is collided

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematic information

    Steps:
        1) Check whether item is collided by agents
        2) Check whether the item-carrying agent is colliding.

    Memo:
        When unloading an item, the agent checks to see if the load will collided with the OBSTACLE or another item before unloading it.
        Therefore, item collidedes only in the above 1) and 2) case
    """
    num_items = env_info.num_items
    # Set array_shape to num_items+1 considering dummy_index
    zeros_frag = jnp.zeros((num_items + 1), dtype=int)

    def _check_discrete_is_item_collided(
        state: State,
        agent_collided: Array,
    ):
        if env_info.is_discrete:
            is_collided_by_agent = jax.vmap(
                lambda item_pos, agent_pos: jnp.any(
                    jnp.all(jnp.equal(agent_pos, item_pos), axis=-1)
                ),
                in_axes=(0, None),
            )(state.item_pos, state.agent_state.pos)
        else:
            is_collided_by_agent = jax.vmap(
                lambda item_pos, agent_pos: jnp.any(
                    jnp.sqrt(jnp.sum((agent_pos - item_pos) ** 2, axis=-1))
                    < (2 * agent_info.rads.flatten())
                ),
                in_axes=(0, None),
            )(state.item_pos, state.agent_state.pos)
        is_collided_by_carrying_agent = jnp.sum(
            jax.vmap(
                lambda load_item_id, agent_collided: zeros_frag.at[load_item_id].set(
                    1 * agent_collided
                )
            )(state.load_item_id, agent_collided),
            axis=0,
        ).astype(bool)[:-1]
        is_collided = jnp.logical_or(
            is_collided_by_agent, is_collided_by_carrying_agent
        )
        return is_collided

    return jax.jit(_check_discrete_is_item_collided)


def _build_compute_is_solved(env_info: EnvInfo, agent_info: AgentInfo):
    def _compute_discrete_is_solved(item_pos: Array, item_goals: Array):
        """
        compute is each item has been delivered to the item goal
        An item is considered to have reached the goal only when it is unloaded at the goal position by the agent

        Args:
            item_pos (Array): item position
            item_goals (Array): item goal

        Attention:
            An item is not considered a goal if it only passes over the item goal.
            However, since position of item being carried is set to INF, the correct calculation can be made by considering only whether the item_pos and goal_pos match.
        """
        return jnp.all(jnp.equal(item_pos, item_goals), axis=-1)

    def _compute_continuous_is_solved(item_pos, item_goals):
        dist = jnp.sqrt(jnp.sum((item_pos - item_goals) ** 2, axis=-1))
        return dist < env_info.item_rads.flatten() * 2

    if env_info.is_discrete:
        return _compute_discrete_is_solved
    else:
        return _compute_continuous_is_solved


def _build_compute_rew_done_info(env_info: EnvInfo):
    num_items = env_info.num_items
    agent_index = jnp.arange(env_info.num_agents)
    decay_duration = env_info.decay_end - env_info.decay_start
    decay_scale = 1 - env_info.min_reward

    def _compute_rew_done_info(
        old_state: State,
        state: State,
        agent_collided: Array,
        item_collided: Array,
        solved: Array,
        trial_info: TrialInfo,
        old_item_starts: Array,
        new_item_starts: Array,
        item_goals: Array,
    ) -> Tuple[Array, Array, TrialInfo]:
        def _compute_solve_rew(
            old_state: State, old_trial_info: TrialInfo, new_trial_info: TrialInfo
        ):
            old_load_item_id = old_state.load_item_id[agent_index]
            is_load_item = old_load_item_id < num_items

            # item solve reward
            solved = (
                new_trial_info.solved[old_load_item_id]
                - old_trial_info.solved[old_load_item_id]
            )

            if env_info.is_decay_reward:
                solve_rew = jnp.clip(
                    1
                    - decay_scale
                    * (state.item_time[old_load_item_id] - env_info.decay_start)
                    / decay_duration,
                    a_min=env_info.min_reward,
                    a_max=1,
                )
                solve_rew = solved * solve_rew * is_load_item
            else:
                solve_rew = solved * env_info.goal_reward * is_load_item

            return solve_rew

        def _compute_distance(
            agent_index,
            old_item_starts,
            new_item_starts,
            item_goals,
            old_load_item_id,
            new_load_item_id,
        ):
            old_item_id = old_load_item_id[agent_index]
            is_unload = (old_load_item_id[agent_index] < num_items) & (
                new_load_item_id[agent_index] >= num_items
            )

            old_distance = jnp.linalg.norm(
                old_item_starts[old_item_id] - item_goals[old_item_id]
            )
            new_distance = jnp.linalg.norm(
                new_item_starts[old_item_id] - item_goals[old_item_id]
            )
            carry_distance = jnp.clip(old_distance - new_distance, a_min=0)
            return carry_distance * is_unload

        def _compute_rew(
            agent_index: Array,
            old_state: State,
            state: State,
            old_trial_info: TrialInfo,
            new_trial_info: TrialInfo,
            old_item_starts: Array,
            new_item_starts: Array,
            item_goals: Array,
        ) -> Array:
            """compute each agent reward to be vmap

            Args:
                agent_index (Array): index of agent
                old_state (State): agent last step state
                state (State): agent current step state
                old_trial_info (TrialInfo): old trial information
                new_trial_info (TrialInfo): new trial information
                old_item_starts (Array): old item start position
                new_item_starts (Array): new item start position
                item_goals (Array): item goal position

            Returns:
                Array: reward for one agent
            """
            solve_rew = _compute_solve_rew(old_state, old_trial_info, new_trial_info)[0]

            # dist reward
            carry_dist = _compute_distance(
                agent_index,
                old_item_starts,
                new_item_starts,
                item_goals,
                old_state.load_item_id,
                state.load_item_id,
            )
            dist_rew = carry_dist * env_info.dist_reward

            # life penalty
            is_life_end = state.life[agent_index] < 0
            life_penalty = is_life_end * env_info.life_penalty

            not_finished = jnp.logical_not(old_trial_info.agent_collided[agent_index])
            return (solve_rew + dist_rew + life_penalty) * not_finished

        agent_collided = jnp.logical_or(agent_collided, trial_info.agent_collided)
        item_collided = jnp.logical_or(item_collided, trial_info.item_collided)

        # goal check
        solved = trial_info.solved + solved.flatten().astype(int)
        solved_time = (
            jnp.minimum(trial_info.solved_time, solved * trial_info.timesteps)
            + (~solved) * jnp.inf
        )
        # is_success = jnp.all(solved)
        is_success = False

        # excess movement
        excess_move = trial_info.excess_move + (state.life < 0).flatten().astype(int)

        # calculate indicator
        delivery_rate = jnp.sum(solved) / env_info.num_agents
        agent_collided_rate = jnp.sum(agent_collided) / env_info.num_agents
        item_collided_rate = jnp.sum(item_collided) / env_info.num_items
        sum_of_cost = jnp.sum(solved_time)
        makespan = jnp.max(solved_time)

        # check timeout
        timesteps = trial_info.timesteps + 1
        timeout = timesteps >= env_info.timeout

        # timeout = jnp.array([is_timeout] * env_info.num_agents)

        new_trial_info = trial_info._replace(
            timesteps=timesteps,
            agent_collided=agent_collided,
            item_collided=item_collided,
            excess_move=excess_move,
            solved=solved,
            solved_time=solved_time,
            timeout=timeout,
            delivery_rate=delivery_rate,
            agent_crash_rate=agent_collided_rate,
            item_crash_rate=item_collided_rate,
            sum_of_cost=sum_of_cost,
            makespan=makespan,
            is_success=is_success,
        )

        rews = jax.vmap(
            _compute_rew, in_axes=(0, None, None, None, None, None, None, None)
        )(
            agent_index,
            old_state,
            state,
            trial_info,
            new_trial_info,
            old_item_starts,
            new_item_starts,
            item_goals,
        )

        # compute done
        item_done = jnp.all(
            jnp.logical_or(new_trial_info.item_collided, new_trial_info.solved)
        )
        env_done = jnp.logical_or(item_done, new_trial_info.timeout)
        done = jnp.logical_or(new_trial_info.agent_collided, env_done)

        return rews, done, new_trial_info

    return jax.jit(_compute_rew_done_info)


def _build_respawn_items(env_info: EnvInfo) -> Callable:
    """respawn finished items

    Args:
        env_info (EnvInfo): environment base information

    Returns:
        Callable: jit-compiled respawn function
    """
    num_items = env_info.num_items
    map_size = env_info.occupancy_map.shape[0]
    is_biased_sample = env_info.is_biased_sample
    biased_mask = (jnp.arange(map_size**2) < (map_size**2 / 2)).astype(bool)

    class Carry(NamedTuple):
        key: jax.random.PRNGKey
        item_pos: Array
        item_goal: Array
        agent_item_obs_map: Array
        done: Array

    def _body_func(i: int, carry: Carry) -> Carry:
        """body function for jax.lax.fori_loop

        Args:
            i (int): loop index
            carry (Carry): information carrier

        Returns:
            Carry: updated information carrier
        """

        def _pass(
            key: PRNGKey,
            all_item_pos: Array,
            all_item_goal: Array,
            agent_item_obs_map: Array,
        ) -> Tuple[PRNGKey, Array, Array, Array]:
            """
            Pass function used for unfinished items.
            This function is an identity map and returns the unchanged input.
            """
            return key, all_item_pos, all_item_goal, agent_item_obs_map

        def _sample(
            key: PRNGKey,
            all_item_pos: Array,
            all_item_goal: Array,
            agent_item_obs_map: Array,
        ) -> Tuple[PRNGKey, Array, Array, Array]:
            """
            sample item start and goal.
            The item is sampled so that there is no overlap with the location of obstacles, agents, or items.
            uppper index respawning item need to consider lower index respawned item positions

            Args:
                key (PRNGKey): random variable key
                all_item_pos (Array): all item positions (this is updated by lower index item respawn)
                all_item_goal (Array): all item goals (this is updated by lower index item respawn)
                agent_item_obs_map (Array): occupancy map considering obstacles, agents, and item locations (this is updated by lower index item respawn)

            Returns:
                Tuple[key,Array, Array, Array]: key, updated_item_pos, updated_item_goal, update_occupancy_map
            """
            key, subkey = jax.random.split(key)
            sample_prob = 1 - agent_item_obs_map.reshape(-1)
            sample_prob = sample_prob / jnp.sum(sample_prob)

            if is_biased_sample:
                start_prob = sample_prob + sample_prob * biased_mask * 10
                start_prob = start_prob / jnp.sum(start_prob)
                item_start = jax.random.choice(
                    subkey, map_size**2, shape=(1,), replace=False, p=start_prob
                )[0]

                goal_prob = sample_prob + sample_prob * (1 - biased_mask) * 10
                goal_prob = goal_prob / jnp.sum(goal_prob)
                item_goal = jax.random.choice(
                    subkey, map_size**2, shape=(1,), replace=False, p=goal_prob
                )[0]
            else:
                item_start, item_goal = jax.random.choice(
                    subkey, map_size**2, shape=(2,), replace=False, p=sample_prob
                )
            item_start = jnp.array(
                [(item_start / map_size).astype(int), item_start % map_size]
            )
            item_goal = jnp.array(
                [(item_goal / map_size).astype(int), item_goal % map_size]
            )
            all_item_pos = all_item_pos.at[i].set(item_start)
            all_item_goal = all_item_goal.at[i].set(item_goal)
            agent_item_obs_map = (
                agent_item_obs_map.at[item_start[0], item_start[1]]
                .set(1)
                .at[item_start[0], item_start[1] - 1]
                .set(1)
            )
            return key, all_item_pos, all_item_goal, agent_item_obs_map

        key, item_pos, item_goal, agent_item_obs_map = jax.lax.cond(
            carry.done[i].astype(bool),
            _sample,
            _pass,
            carry.key,
            carry.item_pos,
            carry.item_goal,
            carry.agent_item_obs_map,
        )
        return Carry(key, item_pos, item_goal, agent_item_obs_map, carry.done)

    def _respawn_item(
        key: PRNGKey, state: State, task_info: TaskInfo, item_done: Array
    ) -> Tuple[PRNGKey, State, TaskInfo]:
        """respawn item start and goal

        Args:
            key (PRNGKey): random variable
            state (State): current environment state
            task_info (TaskInfo): current task information
            item_done (Array): item done. respawn for done=True item

        Returns:
            Tuple[PRNGKey, State, TaskInfo]: new_key, updated_env_state, updated_task_information
        """
        pos = jnp.vstack((state.agent_state.pos, state.item_pos))
        agent_item_obs_map = jnp.sum(
            jax.vmap(
                lambda pos, map: map.at[pos[0], pos[1]].set(1.0),
                in_axes=(0, None),
            )(pos, task_info.obs.occupancy),
            axis=0,
        )
        b = agent_item_obs_map[:, 1:] + agent_item_obs_map[:, :-1]
        agent_item_obs_map = (
            agent_item_obs_map.at[:, :-1].set(b).astype(bool).astype(int)
        )
        carry = Carry(
            key, state.item_pos, task_info.item_goals, agent_item_obs_map, item_done
        )
        carry = jax.lax.fori_loop(0, num_items, _body_func, carry)
        state = state._replace(item_pos=carry.item_pos)
        task_info = task_info._replace(item_goals=carry.item_goal)
        return carry.key, state, task_info

    return jax.jit(_respawn_item)


def _build_check_collide_with_agent_item(env_info: EnvInfo, agent_info: AgentInfo):
    all_rads = jnp.concatenate((agent_info.rads, env_info.item_rads), axis=0)

    def _check_collide_with_agent_item(own_pos: Array, all_pos: Array):
        dist_to_agent_item = (
            jnp.linalg.norm(all_pos - own_pos, axis=1) - all_rads.flatten()
        )
        assert_shape(dist_to_agent_item, (env_info.num_agents + env_info.num_items,))
        return jnp.any(dist_to_agent_item < all_rads.flatten())

    return jax.jit(_check_collide_with_agent_item)


def _build_check_collide_with_obs(agent_info: AgentInfo):
    def _check_collide_with_obs(pos: Array, sdf: Array):
        dist_to_obs = _get_obstacle_dist(pos, sdf)
        return dist_to_obs < agent_info.rads[0][0]

    return jax.jit(_check_collide_with_obs)
