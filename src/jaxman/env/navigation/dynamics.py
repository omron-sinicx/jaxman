from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array
from jaxman.planner.dwa import DWAPlanner

from ..core import AgentInfo, AgentState, EnvInfo
from ..kinematic_dynamics import (
    _build_check_collision_with_agents,
    _build_check_collision_with_obs,
    _build_compute_next_state,
    _get_agent_dist,
    _get_obstacle_dist,
)
from .core import AgentObservation, TaskInfo, TrialInfo
from .observe import _build_observe


# step function
def _build_inner_step(env_info: EnvInfo, agent_info: AgentInfo):
    is_discrete = env_info.is_discrete
    is_diff_drive = env_info.is_diff_drive
    _compute_next_state = _build_compute_next_state(is_discrete, is_diff_drive)
    _compute_rew_done_info = _build_compute_rew_done_info(env_info)
    _check_collision_wiht_agents = _build_check_collision_with_agents(
        env_info, agent_info, is_discrete
    )
    _check_collision_with_obs = _build_check_collision_with_obs(agent_info, is_discrete)
    _get_solve = _build_is_solved(agent_info, is_discrete)
    # if is_discrete:
    #     planner = None
    # else:
    #     planner = DWAPlanner(
    #         compute_next_state=_compute_next_state,
    #         get_obstacle_dist=_get_obstacle_dist,
    #         get_agent_dist=_get_agent_dist,
    #         agent_info=agent_info,
    #     )
    # _observe = _build_observe(env_info, agent_info, planner)

    def _step(
        state: AgentState, actions: Array, task_info: TaskInfo, trial_info: TrialInfo
    ) -> Tuple[AgentState, Array, Array, TrialInfo]:
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

        # observation = _observe(next_state, task_info, jnp.logical_not(done))

        return next_state, rew, done, new_trial_info

    return jax.jit(_step)


def _build_compute_reward(env_info: EnvInfo):
    def _compute_reward(old_trial_info: TrialInfo, new_trial_info: TrialInfo) -> float:
        """
        compute sparse reward given the current environment status.
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

    return jax.jit(_compute_reward)


def _build_compute_rew_done_info(env_info: EnvInfo):
    _compute_reward = _build_compute_reward(env_info)

    def _compute_rew_done_info(
        obs_collided: Array, agent_collided: Array, solved: Array, trial_info: TrialInfo
    ) -> Tuple[float, bool, TrialInfo]:
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
        rew = _compute_reward(trial_info, new_trial_info)

        done = jnp.logical_or(
            jnp.logical_or(new_trial_info.solved, is_timeout), collided
        )

        return rew, done, new_trial_info

    return jax.jit(_compute_rew_done_info)


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
