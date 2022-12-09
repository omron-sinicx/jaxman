"""environment dynamics functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array
from jaxman.planner.dwa import DWAPlanner

from .core import AgentInfo, AgentObservation, AgentState, EnvInfo, TaskInfo, TrialInfo
from .obstacle import ObstacleMap
from .utils import get_scans, xy_to_ij


def _build_observe(
    env_info: EnvInfo,
    agent_info: AgentInfo,
    is_discrete: bool,
    planner: DWAPlanner,
) -> Callable:
    """build observe function

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information
        is_discrete (bool): whether agent action is discrete or not
        planner (DWAPlanner): DWA Planner

    Returns:
        Callable: jit-compiled observe function
    """

    _get_obs_pos = _build_get_obs_pos(env_info, agent_info, is_discrete)
    _get_comm = _build_get_neighbor_communication(env_info)

    def _observe(state: AgentState, task_info: TaskInfo) -> AgentObservation:
        """_observe function

        Args:
            state (AgentState): agent current state
            task_info (TaskInfo): task information

        Returns:
            AgentObservation: agent observation

        Memo:
            obs_info: discrete_space -> obs_map, continuous_space -> edges
        """

        if is_discrete:
            obs_pos = _get_obs_pos(state, task_info.obs.occupancy)
            planner_act = None
        else:
            obs_pos = _get_obs_pos(state, task_info.obs.edges)
            planner_act = planner._act(state, task_info)
        communications = _get_comm(state)

        return AgentObservation(
            state=state,
            goals=task_info.goals,
            scans=obs_pos,
            planner_act=planner_act,
            communications=communications,
        )

    return jax.jit(_observe)


def _build_calc_reward(env_info: EnvInfo):
    def _calc_reward(old_trial_info: TrialInfo, new_trial_info: TrialInfo) -> float:
        """
        Calculate sparse reward given the current environment status.
        If the task is solved, the agent will be rewarded with a positive scalar attenuated by the elapsed time.
        Otherwise, just 0 will be returned.
        Note: this function should be modified to make RL easier

        Args:
            old_trial_info (TrialInfo): current trial's status
            new_trial_info (TrialInfo): next trial's status

        Returns:
            float: reward
        """
        solved = new_trial_info.solved * env_info.goal_reward
        is_collide = new_trial_info.collided_time < jnp.inf
        collide_penalty = is_collide * env_info.crash_penalty

        reward = solved + collide_penalty + env_info.time_penalty  # 0 ~ 1 scalar
        not_finished_list = ~jnp.logical_or(
            old_trial_info.solved, old_trial_info.collided
        )
        reward = reward * not_finished_list
        return reward

    return jax.jit(_calc_reward)


def _build_compute_rew_done_info(env_info: EnvInfo):
    _calc_reward = _build_calc_reward(env_info)

    def _compute_rew_done_info(
        obs_collided: Array, agent_collided: Array, solved: Array, trial_info: TrialInfo
    ) -> tuple[float, bool, TrialInfo]:
        """
        The postprocessing function to summarize step outputs and compute reward, done, and trial_info

        Args:
            obs_collided (Array): is agent collide with obstacles
            obs_collided (Array): is agent collide with other agents
            solved (Array): is agent solve own task
            trial_info (TrialInfo): previous trial info

        Returns:
            tuple[float, bool, TrialInfo]: reward, done, and new trial_info
        """

        obs_collided = jnp.logical_or(obs_collided, trial_info.obs_collided)
        agent_collided = jnp.logical_or(agent_collided, trial_info.agent_collided)
        collided = jnp.logical_or(
            jnp.logical_or(obs_collided, agent_collided), trial_info.collided
        )
        collided_time = (
            jnp.minimum(trial_info.collided_time, collided * trial_info.timesteps)
            + (~collided) * jnp.inf
        )

        # goal check
        solved = jnp.logical_or(trial_info.solved, solved.flatten())
        solved_time = (
            jnp.minimum(trial_info.solved_time, solved * trial_info.timesteps)
            + (~solved) * jnp.inf
        )
        is_success = jnp.all(solved)

        # calculate indicator
        arrival_rate = jnp.sum(solved) / env_info.num_agents
        crash_rate = jnp.sum(collided) / env_info.num_agents
        sum_of_cost = jnp.sum(solved_time)
        makespan = jnp.max(solved_time)

        # check timeout
        timesteps = trial_info.timesteps + 1
        is_timeout = timesteps >= env_info.timeout

        # timeout = jnp.array([is_timeout] * env_info.num_agents)
        timeout = is_timeout & (~jnp.logical_or(solved, collided))

        # create env_info
        new_trial_info = trial_info._replace(
            timesteps=timesteps,
            collided=collided,
            obs_collided=obs_collided,
            agent_collided=agent_collided,
            collided_time=collided_time,
            solved=solved,
            solved_time=solved_time,
            timeout=timeout,
            arrival_rate=arrival_rate,
            crash_rate=crash_rate,
            sum_of_cost=sum_of_cost,
            makespan=makespan,
            is_success=is_success,
        )

        # compute reward
        rew = _calc_reward(trial_info, new_trial_info)

        done = jnp.logical_or(
            jnp.logical_or(new_trial_info.solved, is_timeout), collided
        )

        return rew, done, new_trial_info

    return jax.jit(_compute_rew_done_info)


# step function
def _build_step(
    env_info: EnvInfo, agent_info: AgentInfo, is_discrete: bool, is_diff_drive: bool
):
    _compute_next_state = _build_compute_next_state(is_discrete, is_diff_drive)
    _compute_rew_done_info = _build_compute_rew_done_info(env_info)
    _check_collision_wiht_agents = _build_check_collision_with_agents(
        env_info, agent_info, is_discrete
    )
    _check_collision_with_obs = _build_check_collision_with_obs(agent_info, is_discrete)
    _get_solve = _build_is_solved(agent_info, is_discrete)
    if is_discrete:
        planner = None
    else:
        planner = DWAPlanner(
            compute_next_state=_compute_next_state,
            get_obstacle_dist=_get_obstacle_dist,
            get_agent_dist=_get_agent_dist,
            agent_info=agent_info,
        )
    _observe = _build_observe(env_info, agent_info, is_discrete, planner)

    def _step(
        state: AgentState, actions: Array, task_info: TaskInfo, trial_info: TrialInfo
    ) -> tuple[AgentObservation, Array, Array, TrialInfo]:
        """
        Step function consisting of the following steps:
        1) compute next state
        2) check if any agent collides with obstacles/other agents
        3) check if all agents arrive at their goal
        4) create return vals

         Args:
            state (AgentState): current state
            action (AgentAction): selected action
            task_info(TaskInfo): task information (i.e., goal status)
            trial_info (TrialInfo): trial status

        Returns:
            tuple[AgentObservation, float, bool, TialInfo]: next observation, reward, done, new_trial_info
        """
        not_finished_agents = jnp.logical_not(
            jnp.logical_or(trial_info.solved, trial_info.collided)
        )
        masked_actions = jax.vmap(lambda a, b: a * b)(actions, not_finished_agents)
        possible_next_state = jax.vmap(_compute_next_state)(
            state, masked_actions, agent_info
        )

        obs_collided = _check_collision_with_obs(possible_next_state.pos, task_info.obs)
        agent_collided = _check_collision_wiht_agents(
            state.pos, possible_next_state.pos
        )
        collided = jnp.logical_or(obs_collided, agent_collided)

        solved = _get_solve(possible_next_state.pos, task_info.goals)
        solved = jnp.logical_and(jnp.logical_not(collided), solved)

        rew, done, new_trial_info = _compute_rew_done_info(
            obs_collided, agent_collided, solved, trial_info
        )

        # if agent finish own episode, agent speed is set to 0
        not_finished_agents = jnp.expand_dims(
            jnp.logical_not(jnp.logical_or(collided, solved)), axis=-1
        )
        vel = possible_next_state.vel * not_finished_agents
        ang = possible_next_state.ang * not_finished_agents
        next_state = possible_next_state._replace(vel=vel, ang=ang)

        observation = _observe(next_state, task_info)

        return observation, rew, done, new_trial_info

    return jax.jit(_step)


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
        action = discrete_action_list[action]
        vel = jnp.expand_dims(action[0], -1)
        ang = jnp.expand_dims(action[1], -1)
        next_rot = (state.rot + ang) % 4
        next_pos = state.pos + jnp.array(
            vel
            * jnp.hstack(
                [jnp.sin(next_rot * jnp.pi / 2), jnp.cos(next_rot * jnp.pi / 2)]
            ),
            dtype=int,
        )
        next_state = AgentState(pos=next_pos, rot=next_rot, vel=vel, ang=ang)
        next_state.is_valid()
        return next_state

    if not is_discrete:
        return _compute_continuous_next_state
    elif is_diff_drive:
        return _compute_diff_drive_next_state
    else:
        return _compute_grid_next_state


### solve ###
def _build_is_solved(agent_info: AgentInfo, is_discrete: bool):
    def _discrete_is_solved(pos: Array, goals: Array):
        """
        compute is discrete agent solve its own task.

        Args:
            pos (Array): current agent position
            goals (Array): agent goal

        Returns:
            Array: solved or not
        """
        return jnp.all(jnp.equal(pos, goals), axis=-1)

    def _continuous_is_solved(pos, goals):
        """
        compute is discrete agent solve its own task.

        Args:
            pos (Array): current agent position
            goals (Array): agent goal

        Returns:
            Array: solved or not
        """
        dist_to_goal = jnp.linalg.norm(pos - goals, axis=-1)
        solved = dist_to_goal < agent_info.rads.reshape(-1)
        return solved

    if is_discrete:
        return _discrete_is_solved
    else:
        return _continuous_is_solved


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


### obstacle ditection ###
def _build_get_obs_pos(env_info: EnvInfo, agent_info: AgentInfo, is_discrete: bool):
    if is_discrete:
        _get_obs_pos = _build_get_discrete_obs_pos(env_info)
    else:
        _get_obs_pos = _build_get_continous_obs_pos(env_info, agent_info)
    return _get_obs_pos


def _build_get_discrete_obs_pos(env_info: EnvInfo):
    is_diff_drive = env_info.is_diff_drive
    fov_r = env_info.fov_r
    num_agents = env_info.num_agents
    own_position_map = jnp.zeros((fov_r * 2 + 1, fov_r * 2 + 1)).at[fov_r, fov_r].set(1)

    class Carry(NamedTuple):
        obs_agent_map: Array
        agent_pos: Array

    def _add_agent_pos_to_obs_map(i: int, carry: Carry) -> Carry:
        """add other agent position to obstacle map

        Args:
            i (int): agent index
            carry (Carry): information carry

        Returns:
            Carry: updated carry (i th agent position is added to obstacle map)
        """
        obs_agent_map = carry.obs_agent_map.at[
            carry.agent_pos[i][0] + fov_r, carry.agent_pos[i][1] + fov_r
        ].set(1.0)
        return Carry(obs_agent_map=obs_agent_map, agent_pos=carry.agent_pos)

    def rot90_traceable(m, k):
        k = (k + 2) % 4
        return jax.lax.switch(k, [partial(jnp.rot90, m, k=i) for i in range(4)])

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

    def _get_obs_and_agent_pos(state: AgentState, obs_map: Array) -> Array:
        """
        get flatten neighboring obstacles and agent position

        Args:
            state (AgentState): agent's current state
            obs_map (Array): obstacle map. obs_map is added One padding

        Returns:
            Array: flatten obs and agent position
        """
        obs_map = jnp.pad(obs_map, fov_r, mode="constant", constant_values=1)
        obs_agent_map = jax.lax.fori_loop(
            0,
            num_agents,
            _add_agent_pos_to_obs_map,
            Carry(obs_agent_map=obs_map, agent_pos=state.pos),
        ).obs_agent_map
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


def _build_get_neighbor_communication(env_info: EnvInfo):
    is_discrete = env_info.is_discrete
    is_diff_drive = env_info.is_diff_drive
    num_comm_agents = env_info.num_comm_agents

    def _compute_relative_pos(state: AgentState) -> Array:
        """compute relative position of other agents

        Args:
            state (AgentState): agent's current state

        Returns:
            Array: relative position
        """
        pos = state.pos
        relative_pos = pos - pos[:, None, :]
        if is_diff_drive:
            ang = state.rot.flatten() * jnp.pi / 2 - jnp.pi
        else:
            ang = state.rot.flatten() - jnp.pi
        if (not is_discrete) or is_diff_drive:
            return batched_apply_rotation(relative_pos, ang)
        else:
            return relative_pos

    def _get_comm(state: AgentState) -> Array:
        """get communication from neighboring agent

        Args:
            state (AgentState): agent's current state

        Returns:
            Array: neighboring agent current informations
        """
        relative_pos = _compute_relative_pos(state)
        dist = jnp.sqrt(jnp.sum(relative_pos**2, axis=-1))
        dist_order = jnp.argsort(dist, axis=1)
        dist_order = jax.lax.dynamic_slice_in_dim(
            dist_order, 1, num_comm_agents, axis=1
        )
        neighbor_rel_pos = jax.vmap(lambda rel_pos, index: rel_pos[index])(
            relative_pos, dist_order
        )

        if not is_discrete:
            rot_vel_ang = state.cat()[:, 2:]
            neighbor_rot_vel_ang = jax.vmap(
                lambda rot_vel_ang, index: rot_vel_ang[index], in_axes=[None, 0]
            )(rot_vel_ang, dist_order)
            return jnp.concatenate((neighbor_rel_pos, neighbor_rot_vel_ang), axis=-1)
        elif is_diff_drive:
            rot = state.rot
            neighbor_rot = jax.vmap(lambda rot, index: rot[index], in_axes=[None, 0])(
                rot, dist_order
            )
            return jnp.concatenate((neighbor_rel_pos, neighbor_rot), axis=-1)
        else:
            return neighbor_rel_pos

    return jax.jit(_get_comm)
