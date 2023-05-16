"""vizualize environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

from typing import Union

from chex import Array
from jaxman.env import AgentState, State, TaskInfo, TrialInfo

from .continuous_viz import render_continuous_map, render_simple_continuous_map
from .discrete_viz import render_env


def render_map(
    state: Union[AgentState, State],
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
    trial_info: TrialInfo = None,
    done: Array = None,
    is_discrete: bool = False,
    high_resolution: bool = False,
    task_type: str = "navigation",
):
    if is_discrete:
        img = render_env(
            state,
            goals,
            occupancy,
            trial_info,
            done,
            high_resolution,
            task_type,
        )
    else:
        if high_resolution:
            img = render_continuous_map(state, goals, rads, occupancy, state_traj)
        else:
            img = render_simple_continuous_map(
                state, goals, rads, occupancy, state_traj, task_type
            )
    return img


def render_gif(
    state_traj: Array,
    item_traj: Array,
    goal_traj: Array,
    rads: Array,
    task_info: TaskInfo = None,
    trial_info: TrialInfo = None,
    dones: Array = None,
    is_discrete: bool = False,
    high_quality: bool = False,
    task_type: str = "navigation",
):
    episode_steps = len(state_traj)
    full_obs = []

    for t in range(episode_steps):
        if task_type == "navigation":
            state = AgentState.from_array(state_traj[t])
            goals = task_info.goals
        else:
            state = State.from_array(state_traj[t], item_traj[t])
            goals = goal_traj[t]
        if dones is not None:
            done = dones[t]
        else:
            done = None
        if is_discrete:
            bg = render_env(
                state,
                goals,
                task_info.obs.occupancy,
                trial_info,
                done,
                high_quality,
                task_type,
            )
        else:
            if high_quality:
                bg = render_continuous_map(
                    state,
                    goals,
                    rads,
                    task_info.obs.occupancy,
                    state_traj[t:],
                    task_type,
                )
            else:
                bg = render_simple_continuous_map(
                    state,
                    goals,
                    rads,
                    task_info.obs.occupancy,
                    state_traj[t:],
                    task_type,
                )
        full_obs.append(bg.copy())

    return full_obs
