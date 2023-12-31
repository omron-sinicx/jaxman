"""Definition of rollout dynamics in navigation env

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.kinematic_dynamics import (
    _build_compute_next_state,
    _build_get_relative_positions,
)
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from jaxman.env.navigation.dynamics import _build_inner_step
from jaxman.env.navigation.observe import _build_observe


def _build_compute_agent_intention(
    env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable
) -> Callable:
    num_agents = env_info.num_agents
    is_discrete = env_info.is_discrete
    use_intentions = env_info.use_intentions
    _compute_next_state = _build_compute_next_state(env_info)
    _compute_relative_pos = _build_get_relative_positions(env_info)

    comm_dim = 2  # (rel_pos,) * 2
    if use_intentions:
        comm_dim *= 2

    dummy_comm = jnp.zeros((num_agents, 1, comm_dim))
    dummy_mask = jnp.zeros((num_agents, 1))

    def _compute_agents_intentions(
        observations: AgentObservation,
        actor_params: FrozenDict,
    ) -> Array:
        """
        compute agent intentions. intention represent where agent will move

        Args:
            observations (AgentObservation): current agents' observations
            actor_params (FrozenDict): actor parameters

        Returns:
            Array: intentions
        """
        if use_intentions:
            state = observations.state
            observations = observations.split_observation()
            dummy_observation = observations._replace(
                communication=dummy_comm, agent_mask=dummy_mask
            )
            if is_discrete:
                q_values = actor_fn({"params": actor_params}, dummy_observation)
                actions = jnp.argmax(q_values, axis=-1)
            else:
                actions, _ = actor_fn({"params": actor_params}, dummy_observation)
            next_possible_state = jax.vmap(_compute_next_state)(
                state, actions, agent_info
            )
            intentions = _compute_relative_pos(state, next_possible_state)
        else:
            intentions = jnp.zeros((num_agents, num_agents, 0))
        return intentions

    return jax.jit(_compute_agents_intentions)


def _build_rollout_step(env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable):
    _env_step = _build_inner_step(env_info, agent_info)
    _observe = _build_observe(env_info, agent_info)
    _compute_intentions = _build_compute_agent_intention(env_info, agent_info, actor_fn)

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
        next_observatinos = _observe(state, task_info, jnp.logical_not(dones))
        next_intentions = _compute_intentions(next_observatinos, actor_params)
        next_observatinos = next_observatinos._replace(intentions=next_intentions)

        return next_state, next_observatinos, rews, dones, new_trial_info

    return jax.jit(_step)
