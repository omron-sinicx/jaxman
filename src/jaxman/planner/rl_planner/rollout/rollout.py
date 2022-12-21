"""jax jit-compiled rollout function

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo, TaskInfo, TrialInfo
from jaxman.env.dynamics import (
    _build_compute_next_state,
    _build_observe,
    _get_agent_dist,
    _get_obstacle_dist,
)
from jaxman.env.instance import Instance
from jaxman.env.obstacle import ObstacleMap
from jaxman.env.task_generator import sample_valid_start_goal
from jaxman.planner.dwa import DWAPlanner
from jaxman.planner.rl_planner.agent.sample_action import build_sample_agent_action
from jaxman.planner.rl_planner.memory.dataset import Experience

from .dynamics import _build_compute_agent_intention, _build_rollout_step


class Carry(NamedTuple):
    episode_steps: int
    state: AgentState
    task_info: TaskInfo
    trial_info: TrialInfo
    observations: Array
    key: PRNGKey
    experience: Experience
    rewards: Array
    dones: Array

    @classmethod
    def reset(
        self,
        env_info: EnvInfo,
        agent_info: AgentInfo,
        obs: ObstacleMap,
        key: PRNGKey,
        actor_params: FrozenDict,
        _observe: Callable,
        _compute_intentions: Callable,
    ):
        episode_steps = 0
        subkey, key = jax.random.split(key)

        num_agents = env_info.num_agents
        starts, start_rots, goals = sample_valid_start_goal(
            subkey,
            agent_info.rads,
            obs,
            num_agents,
            env_info.is_discrete,
            env_info.is_discrete,
        )
        task_info = TaskInfo(starts, start_rots, goals, obs)
        state = AgentState(
            pos=task_info.starts,
            rot=task_info.start_rots,
            vel=jnp.zeros_like(task_info.start_rots),
            ang=jnp.zeros_like(task_info.start_rots),
        )
        observations = _observe(state, task_info, jnp.ones((num_agents,), dtype=bool))
        trial_info = TrialInfo().reset(env_info.num_agents)
        rewards = jnp.zeros((num_agents,))
        dones = jnp.array([False] * num_agents)

        intentions = _compute_intentions(observations, actor_params)
        observations = observations._replace(intentions=intentions)

        if env_info.is_discrete:
            actions = jnp.zeros((num_agents,))
        else:
            actions = jnp.zeros((num_agents, 2))
        experience = Experience.reset(
            num_agents,
            env_info.timeout,
            observations.cat(),
            actions,
        )
        return self(
            episode_steps,
            state,
            task_info,
            trial_info,
            observations,
            key,
            experience,
            rewards,
            dones,
        )


def build_rollout_episode(
    instance: Instance,
    actor_fn: Callable,
    evaluate: bool,
) -> Callable:
    """build rollout episode function

    Args:
        instance (Instance): problem instance
        actor_fn (Callable): actor function
        evaluate (bool): whether agent explorate or evaluate

    Returns:
        Callable: jit-compiled rollout episode function
    """
    env_info, agent_info, _ = instance.export_info()

    _step = _build_rollout_step(
        env_info,
        agent_info,
        actor_fn,
    )
    if instance.is_discrete:
        planner = None
    else:
        compute_next_state = _build_compute_next_state(
            instance.is_discrete, instance.is_diff_drive
        )
        planner = DWAPlanner(
            compute_next_state=compute_next_state,
            get_obstacle_dist=_get_obstacle_dist,
            get_agent_dist=_get_agent_dist,
            agent_info=agent_info,
        )
    _observe = _build_observe(env_info, agent_info, env_info.is_discrete, planner)
    _compute_intentions = _build_compute_agent_intention(env_info, agent_info, actor_fn)
    _sample_actions = build_sample_agent_action(
        actor_fn, instance.is_discrete, evaluate
    )

    def _rollout_episode(
        key: PRNGKey,
        actor_params: FrozenDict,
        obs: ObstacleMap,
        carry: Carry = None,
    ):
        if not carry:
            carry = Carry.reset(
                env_info,
                agent_info,
                obs,
                key,
                actor_params,
                _observe,
                _compute_intentions,
            )

        def _act_and_step(carry: Carry):
            not_finished_agent = ~carry.dones
            key, actions = _sample_actions(actor_params, carry.observations, carry.key)
            actions = jax.vmap(lambda action, mask: action * mask)(
                actions, not_finished_agent
            )
            next_observations, rews, dones, new_trial_info = _step(
                carry.state, actions, carry.task_info, carry.trial_info, actor_params
            )
            next_state = next_observations.state

            rewards = carry.rewards + rews
            experience = carry.experience.push(
                carry.episode_steps, carry.observations.cat(), actions, rews, dones
            )

            carry = Carry(
                episode_steps=carry.episode_steps + 1,
                state=next_state,
                task_info=carry.task_info,
                trial_info=new_trial_info,
                observations=next_observations,
                key=key,
                experience=experience,
                rewards=rewards,
                dones=dones,
            )
            return carry

        def cond(carry):
            return jnp.logical_not(jnp.all(carry.dones))

        carry = jax.lax.while_loop(cond, _act_and_step, carry)
        return carry

    return jax.jit(_rollout_episode)
