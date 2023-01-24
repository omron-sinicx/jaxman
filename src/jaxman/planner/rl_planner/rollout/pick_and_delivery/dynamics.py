"""Definition of rollout dynamics in pick and delivery env

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.kinematic_dynamics import _build_get_relative_positions
from jaxman.env.pick_and_delivery.core import (
    AgentObservation,
    State,
    TaskInfo,
    TrialInfo,
)
from jaxman.env.pick_and_delivery.dynamics import (
    _build_compute_next_env_state,
    _build_inner_step,
)
from jaxman.env.pick_and_delivery.observe import _build_observe
from jaxman.planner.rl_planner.core import AgentObservation as ModelInput


def _build_compute_agent_intention(
    env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable
) -> Callable:
    num_agents = env_info.num_agents
    is_discrete = env_info.is_discrete
    is_diff_drive = env_info.is_diff_drive
    _compute_next_state = _build_compute_next_env_state(env_info, agent_info)
    _compute_relative_pos = _build_get_relative_positions(env_info)

    if not is_discrete:
        comm_dim = 10  # (rel_pos, rot, vel, ang) * 2
    elif env_info.is_diff_drive:
        comm_dim = 6  # (rel_pos, rot) * 2
    else:
        comm_dim = 4  # (rel_pos,) * 2

    dummy_comm = jnp.zeros((num_agents, 1, comm_dim))
    dummy_mask = jnp.zeros((num_agents, 1))

    def _compute_agents_intentions(
        state: State,
        observations: AgentObservation,
        actor_params: FrozenDict,
        task_info: TaskInfo,
    ) -> Array:
        """
        compute agent intentions. intention represent where agent will move

        Args:
            observations (AgentObservation): current agents' observations
            actor_params (FrozenDict): actor parameters

        Returns:
            Array: intentions
        """
        observations = observations.split_observation()
        dummy_observation = observations._replace(
            communication=dummy_comm, agent_mask=dummy_mask
        )

        if is_discrete:
            action_probs = actor_fn({"params": actor_params}, dummy_observation)
            actions = jnp.argmax(action_probs, axis=-1)
        else:
            means, _ = actor_fn({"params": actor_params}, dummy_observation)
            actions = means

        # compute relative position
        next_possible_state = _compute_next_state(state, actions, task_info)

        intentions = _compute_relative_pos(
            state.agent_state, next_possible_state.agent_state
        )
        return intentions

    return jax.jit(_compute_agents_intentions)


def _build_rollout_step(env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable):
    _env_step = _build_inner_step(env_info, agent_info)
    _compute_intentions = _build_compute_agent_intention(env_info, agent_info, actor_fn)
    _observe = _build_observe(env_info, agent_info)

    def _step(
        state: AgentState,
        actions: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
        actor_params: FrozenDict,
    ) -> Tuple[AgentObservation, Array, Array, TrialInfo]:
        """
        compute next step based on environmental step
        In addition to the environmental step function, this step function calculates the intention and stores it in AgentObservation.

        Args:
            state (AgentState): current state
            actions (AgentAction): selected action
            task_info(TaskInfo): task information (i.e., goal status)
            trial_info (TrialInfo): trial status
            actor_params (FrozenDict): actor parameters

        Returns:
            Tuple[AgentObservation, Array, Array, TrialInfo]: next observations, rewards, dones, new_trial_info
        """
        next_state, rews, dones, new_trial_info = _env_step(
            state, actions, task_info, trial_info
        )
        next_observations = _observe(next_state, task_info, new_trial_info)
        next_intentions = _compute_intentions(
            next_state, next_observations, actor_params, task_info
        )
        next_observations = next_observations._replace(intentions=next_intentions)

        return next_state, next_observations, rews, dones, new_trial_info

    return jax.jit(_step)
