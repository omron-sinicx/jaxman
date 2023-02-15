"""observation function for navigation environment

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
    _build_get_obs_pos,
    _build_get_other_agent_infos,
)
from ..core import AgentInfo, AgentState, EnvInfo
from .core import AgentObservation, TaskInfo


def _build_observe(
    env_info: EnvInfo,
    agent_info: AgentInfo,
) -> Callable:
    """build observe function

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information

    Returns:
        Callable: jit-compiled observe function
    """
    _get_obs_pos = _build_get_obs_pos(env_info, agent_info)
    _get_other_agent_infos = _build_get_other_agent_infos(env_info)
    _get_mask = _build_compute_neighbor_mask(env_info)
    _planner = create_planner(env_info, agent_info)

    def _observe(state: AgentState, task_info: TaskInfo, not_finished_agent: Array):
        if env_info.is_discrete:
            obs_pos = _get_obs_pos(state, task_info.obs.occupancy)
            planner_act = None
        else:
            obs_pos = _get_obs_pos(state, task_info.obs.edges)
            planner_act = _planner._act(state, task_info)
        relative_positions = _get_other_agent_infos(state, state)
        intentions = jnp.zeros_like(relative_positions)
        masks = _get_mask(relative_positions[:, :, :2], not_finished_agent)
        return AgentObservation(
            state=state,
            goals=task_info.goals,
            scans=obs_pos,
            planner_act=planner_act,
            relative_positions=relative_positions,
            intentions=intentions,
            masks=masks,
        )

    return jax.jit(_observe)
