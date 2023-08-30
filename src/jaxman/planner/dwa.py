"""Classical planner. Dynamic window approach

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jaxman.env import AgentInfo, AgentState, EnvInfo, TaskInfo
from jaxman.env.kinematic_dynamics import (
    _build_compute_next_state,
    _get_agent_dist,
    _get_obstacle_dist,
)
from jaxman.utils import standardize


def create_planner(env_info: EnvInfo, agent_info: AgentInfo, is_individual_plannning = False):
    _compute_next_state = _build_compute_next_state(env_info)
    return DWAPlanner(
        compute_next_state=_compute_next_state,
        get_obstacle_dist=_get_obstacle_dist,
        get_agent_dist=_get_agent_dist,
        agent_info=agent_info,
        use_acc=env_info.use_acc,
        is_individual_planning = is_individual_plannning,
    )


@dataclass
class DWAPlanner:
    compute_next_state: Callable
    get_obstacle_dist: Callable
    get_agent_dist: Callable
    agent_info: AgentInfo
    use_acc: bool
    velocity_resolution: int = 8
    angular_velocity_resolution: int = 8
    decay_temperature: float = 1e-1
    distance_threshold: float = 1e-1
    heading_weight: float = 1.0
    velocity_weight: float = 1.0
    distance_weight: float = 1.0
    goal_weight: float = 10.0
    is_individual_planning: bool = False

    def __post_init__(self):
        self._act = self.build_act()

    def build_act(self) -> None:
        # to be vmapped
        def _act(
            state: AgentState,
            goals: Array,
            sdf_map: Array,
            agent_info: AgentInfo,
            self_id: int,
            all_state: AgentState,
        ) -> Array:

            diff = goals - state.pos  # (2, )
            goal_angle = jnp.arctan2(diff[0], diff[1]) % (2 * jnp.pi)  # float
            goal_dist = jnp.linalg.norm(diff)  # float

            goal_based_decay = jnp.exp(-goal_dist / self.decay_temperature)

            act0 = jnp.linspace(
                -1,
                (1 - goal_based_decay),
                self.velocity_resolution,
            ).flatten()  # (acceleration_resolution, )
            act1 = jnp.linspace(
                -1,
                1,
                self.angular_velocity_resolution,
            ).flatten()  # (angular_acceleration_resolution, 1)
            # (v_res x av_res, 2)
            dwin = jnp.vstack([x.flatten() for x in jnp.meshgrid(act0, act1)]).T

            if self.use_acc:
                next_vel = jnp.clip(
                    state.vel + dwin[:, 0] * agent_info.max_accs,
                    a_min=0,
                    a_max=agent_info.max_vels,
                )
                next_ang_vel = jnp.clip(
                    state.vel + dwin[:, 1] * agent_info.max_ang_accs,
                    a_min=-agent_info.max_ang_vels,
                    a_max=agent_info.max_ang_vels,
                )
                angle_diff = jnp.abs(
                    state.rot + next_ang_vel - goal_angle
                )  # (v_res x av_res, )
                angle_diff = jnp.minimum(angle_diff, jnp.pi * 2 - angle_diff)
                heading_score = jnp.pi - angle_diff
                vel_score = next_vel  # (v_res x av_res, )
            else:
                angle_diff = jnp.abs(
                    state.rot + dwin[:, 1] * agent_info.max_ang_vels - goal_angle
                )
                angle_diff = jnp.minimum(angle_diff, jnp.pi * 2 - angle_diff)
                heading_score = jnp.pi - angle_diff
                vel_score = dwin[:, 0] * agent_info.max_vels  # (v_res x av_res, )

            next_states = jax.vmap(
                partial(self.compute_next_state, state=state, agent_info=agent_info)
            )(
                actions=dwin
            )  # (v_res x av_res, 2 or 1)

            obs_dist_score = jax.vmap(partial(self.get_obstacle_dist, sdf_map=sdf_map))(
                pos=next_states.pos,
            )  # (v_res x av_res, )
            # (v_res x av_res)

            # ignore the existence of other agents
            if self.is_individual_planning:
                dist_score = obs_dist_score
            else:
                agent_dist_score = jax.vmap(
                    partial(
                        self.get_agent_dist,
                        all_pos=all_state.pos,
                        agent_info=agent_info,
                        self_id=self_id,
                    )
                )(query_pos=next_states.pos).min(axis=1)
                dist_score = jnp.minimum(obs_dist_score, agent_dist_score)
            dist_score = jnp.minimum(dist_score, self.distance_threshold)
            # (v_res x av_res, )

            goal_dist = jnp.linalg.norm(next_states.pos - goals, axis=1)
            goal_score = 1 - goal_dist

            score = (
                standardize(heading_score) * self.heading_weight  # * goal_based_decay
                + standardize(vel_score) * self.velocity_weight
                + standardize(dist_score) * self.distance_weight
                + standardize(goal_score) * self.goal_weight
            )  # (v_res x av_res, )

            valid = dist_score > agent_info.rads  # (v_res x av_res, )
            score = jax.vmap(
                lambda s, v: jax.lax.cond(v, lambda _: s, lambda _: -jnp.inf, None)
            )(
                score, valid
            )  # (v_res x av_res, )

            action = dwin[jnp.argmax(score)]

            return action

        def act(state: AgentState, goals: Array, sdf: Array) -> Array:
            """
            Plan next step based on the current state and task information

            Args:
                state (AgentState): current state
                goal (Array): goal potitions
                sdf (Array): sdf map

            Returns:
                AgentAction: next action
            """
            return jax.vmap(_act, in_axes=(0, 0, None, 0, 0, None))(
                state,
                goals,
                sdf,
                self.agent_info,
                jnp.arange(state.pos.shape[0]),
                state,
            )

        return jax.jit(act)
