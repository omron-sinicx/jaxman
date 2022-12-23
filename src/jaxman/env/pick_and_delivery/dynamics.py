from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from chex import Array
from jaxman.utils import compute_distance

from ..core import AgentInfo, AgentState, EnvInfo
from ..kinematic_dynamics import (
    _build_check_collision_with_agents,
    _build_check_collision_with_obs,
    _build_compute_next_state,
)
from ..obstacle import ObstacleMap
from .core import AgentObservation, State, TaskInfo, TrialInfo

INF = 10000


def _build_inner_step(env_info: EnvInfo, agent_info: AgentInfo):
    is_discrete = env_info.is_discrete
    _compute_next_state = _build_compute_next_env_state(env_info, agent_info)
    _check_collision_wiht_agents = _build_check_collision_with_agents(
        env_info, agent_info, is_discrete
    )
    _check_collision_with_obs = _build_check_collision_with_obs(agent_info, is_discrete)
    _check_collision_with_items = _build_check_collision_with_items(
        is_discrete, agent_info
    )
    _compute_is_item_solved = _build_compute_is_solved(is_discrete, agent_info)
    _check_is_item_collided = _build_check_is_item_collided(env_info, agent_info)
    _compute_rew_done_info = _build_compute_rew_done_info(env_info)

    def _inner_step(
        state: State, actions: Array, task_info: TaskInfo, trial_info: TrialInfo
    ) -> Tuple[AgentObservation, Array, Array, TrialInfo]:

        # update state
        movable_agents = jnp.logical_not(trial_info.agent_collided)
        masked_actions = jax.vmap(lambda a, b: a * b)(actions, movable_agents)
        possible_next_state = _compute_next_state(state, masked_actions, task_info)

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
        solved = _compute_is_item_solved(
            possible_next_state.item_pos, task_info.item_goals
        )
        item_collided = _check_is_item_collided(possible_next_state, agent_collided)

        rews, done, new_trial_info = _compute_rew_done_info(
            state,
            possible_next_state,
            agent_collided,
            item_collided,
            solved,
            task_info,
            trial_info,
        )

        # if agent finish own episode, agent speed is set to 0
        not_finished_agents = jnp.expand_dims(jnp.logical_not(agent_collided), axis=-1)
        vel = possible_next_state.agent_state.vel * not_finished_agents
        ang = possible_next_state.agent_state.ang * not_finished_agents
        new_agent_state = possible_next_state.agent_state._replace(vel=vel, ang=ang)

        # if item finish episode, set item position to INF
        done_item = jnp.logical_or(item_collided, solved)
        item_pos = jax.vmap(lambda pos, done: pos + done * INF)(
            possible_next_state.item_pos, done_item
        )

        next_state = possible_next_state._replace(
            agent_state=new_agent_state, item_pos=item_pos
        )

        return next_state, rews, done, new_trial_info

    return jax.jit(_inner_step)


def _build_compute_next_env_state(env_info: EnvInfo, agent_info: AgentInfo):
    _compute_agent_next_state = _build_compute_next_agent_state(env_info)
    _compute_next_item_state = _build_compute_next_item_state(env_info, agent_info)

    def _compute_next_env_state(
        state: State, actions: Array, task_info: TaskInfo
    ) -> State:
        next_agent_state = _compute_agent_next_state(
            state.agent_state, actions, agent_info
        )
        next_load_item_id, next_item_pos = _compute_next_item_state(
            next_agent_state, actions, state.load_item_id, state.item_pos, task_info
        )
        return State(next_agent_state, next_load_item_id, next_item_pos)

    return jax.jit(_compute_next_env_state)


def _build_compute_next_agent_state(env_info: EnvInfo):
    _compute_next_state = _build_compute_next_state(
        env_info.is_discrete, env_info.is_diff_drive
    )

    def _apply_load_or_unload(
        current_state: State, next_possible_state: AgentState, action: Array
    ):
        can_move = action < 5
        arrayed_next_state = (
            can_move * next_possible_state.cat() + ~can_move * current_state.cat()
        )
        if env_info.is_discrete:
            arrayed_next_state = arrayed_next_state.astype(int)
        return AgentState.from_array(arrayed_next_state)

    def _compute_agent_next_state(state: AgentState, actions: Array, agent_info):
        possible_next_state = jax.vmap(_compute_next_state)(state, actions, agent_info)
        next_state = jax.vmap(_apply_load_or_unload)(
            state, possible_next_state, actions
        )
        return next_state

    return jax.jit(_compute_agent_next_state)


def _build_compute_next_item_state(env_info: EnvInfo, agent_info: AgentInfo):
    num_items = env_info.num_items
    dummy_index = num_items
    num_agents = env_info.num_agents

    class Carry(NamedTuple):
        state: AgentState
        actions: Array
        load_item_id: int
        item_pos: Array
        obs: ObstacleMap

    def _move_or_load_unload_items(i: int, carry: Carry):
        """compute item next state to be jax.lax.for_i_loop

        Args:
            i (int): for_loop index
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
            def unload_items(i: int, carry: Carry):
                # grid case
                possible_item_pos = carry.state.pos[i] + jnp.array([0, -1], dtype=int)
                # Check whether there are any obstacles at the location to unload
                obstacle_collide = jnp.array(
                    carry.obs.occupancy[possible_item_pos[0], possible_item_pos[1]],
                    dtype=bool,
                )
                # Check whether there are any items at the location to unload
                item_collide = jnp.any(
                    jnp.all(jnp.equal(carry.item_pos, possible_item_pos), axis=-1)
                )

                can_unload = jnp.logical_not(
                    jnp.logical_or(obstacle_collide, item_collide)
                )

                # if agent can unload items, item_position is updated, else item position is set to INF (agent continue to load item)
                next_item_pos = carry.item_pos.at[carry.load_item_id[i]].set(
                    can_unload * possible_item_pos + ~can_unload * jnp.array([INF, INF])
                )
                # if agent can unload items, load_item_id for i th agent is set to dummy_index(dot't carry any item)
                next_load_item_id = carry.load_item_id.at[i].set(
                    can_unload * dummy_index + ~can_unload * carry.load_item_id[i]
                )

                return carry._replace(
                    item_pos=next_item_pos, load_item_id=next_load_item_id
                )

            def load_items(i: int, carry: Carry):
                dist_to_item = jnp.sum(
                    (carry.item_pos - carry.state.pos[i]) ** 2, axis=1
                )
                nearest_item_index = jnp.argmin(dist_to_item)
                nearest_item_dist = dist_to_item[nearest_item_index]
                # discrete case
                can_pickup = nearest_item_dist <= 1
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

        is_move = carry.actions[i] < 5
        return jax.lax.cond(is_move, move_agent, load_or_unload, i, carry)

    def _compute_next_item_state(
        state: AgentState,
        actions: Array,
        load_item_id: Array,
        item_pos: Array,
        task_info: TaskInfo,
    ) -> Tuple[Array, Array]:
        carry = Carry(state, actions, load_item_id, item_pos, task_info.obs)
        carry = jax.lax.fori_loop(0, num_agents, _move_or_load_unload_items, carry)
        return carry.load_item_id, carry.item_pos

    return jax.jit(_compute_next_item_state)


def _build_check_collision_with_items(is_discrete: bool, agent_info: AgentInfo):
    def _check_discrete_collision_with_items(pos: Array, item_pos: Array):
        collide = jax.vmap(
            lambda pos, item_pos: jnp.any(jnp.all(jnp.equal(item_pos, pos), axis=-1)),
            in_axes=(0, None),
        )(pos, item_pos)
        return collide

    def _check_continuous_collision_with_items(pos: Array, item_pos: Array):
        pass

    if is_discrete:
        return _check_discrete_collision_with_items
    else:
        return _check_continuous_collision_with_items


def _build_check_is_item_collided(env_info: EnvInfo, agent_info: AgentInfo):
    """check whether item is collided

    Args:
        is_discrete (bool): is_discrete
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
        is_collided_by_agent = jax.vmap(
            lambda item_pos, agent_pos: jnp.any(
                jnp.all(jnp.equal(agent_pos, item_pos), axis=-1)
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


def _build_compute_is_solved(is_discrete: bool, agent_info: AgentInfo):
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
        pass

    if is_discrete:
        return _compute_discrete_is_solved
    else:
        return _compute_continuous_is_solved


def _build_compute_rew_done_info(env_info: EnvInfo):
    num_items = env_info.num_items
    agent_index = jnp.arange(env_info.num_agents)

    def _compute_rew_done_info(
        old_state: State,
        state: State,
        agent_collided: Array,
        item_collided: Array,
        solved: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
    ) -> Tuple[Array, Array, TrialInfo]:
        def _compute_rew(
            agent_index: Array,
            old_state: State,
            state: State,
            task_info: TaskInfo,
            old_trial_info: TrialInfo,
            new_trial_info: TrialInfo,
        ) -> Array:
            """compute each agent reward to be vmap

            Args:
                agent_index (Array): index of agent
                old_state (State): agent last step state
                state (State): agent current step state
                old_trial_info (TrialInfo): old trial information
                new_trial_info (TrialInfo): new trial information

            Returns:
                Array: reward for one agent
            """
            old_load_item_id = old_state.load_item_id[agent_index]
            is_load_item = old_load_item_id < num_items

            # item solve reward
            solved = new_trial_info.solved[old_load_item_id]
            solve_rew = solved * env_info.goal_reward * is_load_item

            # item distance reward
            old_distance = compute_distance(
                task_info.item_goals[old_load_item_id],
                old_state.agent_state.pos[agent_index],
            )
            current_distance = compute_distance(
                task_info.item_goals[old_load_item_id],
                state.agent_state.pos[agent_index],
            )
            dist_rew = (
                (old_distance - current_distance) * env_info.dist_reward * is_load_item
            )

            # collide penalty
            crash_penalty = (
                new_trial_info.agent_collided[agent_index] * env_info.crash_penalty
            )

            # dont hold item penalty
            load_item_id = state.load_item_id[agent_index]
            is_load_item = load_item_id < num_items
            dont_hold_item_penalty = (
                jnp.logical_not(is_load_item) * env_info.dont_hold_item_penalty
            )

            not_finished = jnp.logical_not(old_trial_info.agent_collided[agent_index])
            return (
                solve_rew + dist_rew + crash_penalty + dont_hold_item_penalty
            ) * not_finished

        agent_collided = jnp.logical_or(agent_collided, trial_info.agent_collided)
        item_collided = jnp.logical_or(item_collided, trial_info.item_collided)

        # goal check
        solved = jnp.logical_or(trial_info.solved, solved.flatten())
        solved_time = (
            jnp.minimum(trial_info.solved_time, solved * trial_info.timesteps)
            + (~solved) * jnp.inf
        )
        is_success = jnp.all(solved)

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

        rews = jax.vmap(_compute_rew, in_axes=(0, None, None, None, None, None))(
            agent_index, old_state, state, task_info, trial_info, new_trial_info
        )

        # compute done
        item_done = jnp.all(
            jnp.logical_or(new_trial_info.item_collided, new_trial_info.solved)
        )
        env_done = jnp.logical_or(item_done, new_trial_info.timeout)
        done = jnp.logical_or(new_trial_info.agent_collided, env_done)

        return rews, done, new_trial_info

    return jax.jit(_compute_rew_done_info)
