"""sampling agent start and goals

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

from .obstacle import ObstacleMap

diff_drive_rots = jnp.array([0.0, jnp.pi / 2, jnp.pi, jnp.pi * 3 / 2])


@partial(jax.jit, static_argnames=("num_samples", "is_discrete", "is_diff_drive"))
def sample_start_rots(
    key: PRNGKey, num_samples: int, is_discrete: bool, is_diff_drive: bool
) -> Array:
    """sample agent start rotations

    Args:
        key (PRNGKey): random key variable
        num_samples (int): number to sample
        is_discrete (bool): whether environment is discrete space or not
        is_diff_drive (bool): whether environment is diff drive env or not

    Returns:
        Array: sample agent start rotations
    """
    if is_discrete:
        if is_diff_drive:
            start_rots = jax.random.choice(key, 4, shape=(num_samples, 1))
        else:
            start_rots = jnp.zeros((num_samples, 1))
    else:
        start_rots = jax.random.uniform(key, shape=(num_samples, 1)) * 2 * jnp.pi
    return start_rots


@partial(
    jax.jit,
    static_argnames=("num_agents", "num_max_trials", "is_discrete", "sample_type"),
)
def sample_random_pos(
    key: PRNGKey,
    rads: Array,
    obs: ObstacleMap,
    num_agents: int,
    is_discrete: bool,
    no_overlap: bool = True,
    sample_type: str = "uniform",
    num_max_trials: bool = 200,
) -> Tuple[Array, Array]:
    """sample agent start and goal position

    Args:
        key (PRNGKey): random key variables
        rads (Array): agent radius
        obs (ObstacleMap): obstacle map
        num_agents (int): number of agent
        is_discrete (bool): whether environment is discrete space or not
        no_overlap (bool, optional): whether to allow overlap in sampled positions. Defaults to True.
        sample_type (str, optional): sample type. Defaults to "uniform".
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Tuple[Array, Array]: sampled agent start and goal
    """
    if sample_type == "uniform":
        starts, goals = sample_uniform(
            key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            no_overlap,
            num_max_trials,
        )
    else:
        starts, goals = sample_from_corner(
            key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            no_overlap,
            num_max_trials,
        )
    if is_discrete:
        map_size = obs.sdf.shape[0]
        starts = jnp.minimum((starts * map_size).astype(int), map_size - 1)
        goals = jnp.minimum((goals * map_size).astype(int), map_size - 1)
    return starts, goals


@partial(jax.jit, static_argnames=("num_samples", "num_max_trials", "is_discrete"))
def sample_uniform(
    key: PRNGKey,
    num_samples: int,
    rads: Array,
    obs: ObstacleMap,
    is_discrete: bool,
    no_overlap: bool = False,
    num_max_trials: int = 100,
) -> Array:
    """
    Sample a list of vertices from free area

    Args:
        key (PRNGKey): jax.random.PRNGKey
        num_samples (int): number of vertices to sample
        rads (Array): agent's radius
        obs (ObstacleMap): signed distance function of the map
        is_discrete (bool): whether environment action space is discrete or not
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Array: list of random valid positions
    """

    if rads.size == 1:
        rads = jnp.ones(num_samples) * rads
    key0, key1, key2, key3 = jax.random.split(key, 4)
    carried_start = jax.random.uniform(key0, (num_samples, 2))
    carried_goal = jax.random.uniform(key1, (num_samples, 2))

    loop_carry = [key2, carried_start, rads, obs]
    starts = jax.lax.fori_loop(
        0,
        num_samples,
        partial(
            _sample_random_pos,
            is_discrete=is_discrete,
            no_overlap=no_overlap,
            num_max_trials=num_max_trials,
        ),
        loop_carry,
    )[1]

    loop_carry = [key3, carried_goal, rads, obs]
    goals = jax.lax.fori_loop(
        0,
        num_samples,
        partial(
            _sample_random_pos,
            is_discrete=is_discrete,
            no_overlap=no_overlap,
            num_max_trials=num_max_trials,
        ),
        loop_carry,
    )[1]
    return starts, goals


def _sample_random_pos(
    i: int,
    loop_carry: list,
    is_discrete: bool,
    no_overlap: bool = False,
    num_max_trials: int = 100,
) -> list:
    """Compiled function of sample_random_pos"""

    def cond(while_carry):
        (
            target_pos,
            target_rad,
            obs,
            carreid_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry[1:]
        map_size = obs.sdf.shape[0]
        pos_int = jnp.minimum((target_pos * map_size).astype(int), map_size - 1)
        if is_discrete:
            obs_overlap = jnp.array(obs.occupancy[pos_int[0], pos_int[1]], dtype=bool)
            carry_int = jnp.minimum((carried_pos * map_size).astype(int), map_size - 1)
            agent_overlap = jnp.any(jnp.all(jnp.equal(pos_int, carry_int), axis=-1))
            return obs_overlap | agent_overlap
        else:
            return (
                jnp.any(
                    jnp.linalg.norm(carreid_pos - target_pos, axis=1)
                    - target_rad * 2.5
                    - rads * 2.5
                    < 0
                )
                & no_overlap
            ) | (obs.sdf[pos_int[0], pos_int[1]] - target_rad * 1.5 < 0)

    def body(while_carry):
        (
            key,
            target_pos,
            target_rad,
            obs,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry
        num_trials = num_trials + 1
        key0, key = jax.random.split(key)
        target_pos = (
            jax.random.uniform(key0, shape=(2,)) * (1 - 2 * target_rad) + target_rad
        )
        return [
            key,
            target_pos,
            target_rad,
            obs,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ]

    key, carried_pos, rads, obs = loop_carry
    num_trials = 0
    target_pos = carried_pos[i]
    target_rad = rads[i]

    while_carry = [
        key,
        target_pos,
        target_rad,
        obs,
        carried_pos,
        rads,
        no_overlap,
        num_trials,
    ]

    while_carry = jax.lax.while_loop(cond, body, while_carry)
    key = while_carry[0]
    pos = while_carry[1]
    num_trials = while_carry[-1]
    # pos = jax.lax.cond(
    #     num_trials < num_max_trials, lambda _: pos, lambda _: pos * jnp.inf, None
    # )
    carried_pos = carried_pos.at[i].set(pos)
    return [key, carried_pos, rads, obs]


@partial(jax.jit, static_argnames=("num_agent", "num_max_trials", "is_discrete"))
def sample_from_corner(
    key: PRNGKey,
    num_agent: int,
    rads: Array,
    obs: ObstacleMap,
    is_discrete: bool,
    no_overlap: bool = False,
    num_max_trials: int = 100,
) -> Array:
    """
    Sample a list of vertices from free area

    Args:
        key (PRNGKey): jax.random.PRNGKey
        num_agent (int): number of vertices to sample
        rads (Array): agent's radius
        obs (ObstacleMap): obstacle map
        is_discrete (bool): whether environment action space is discrete or not
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Array: list of random valid positions
    """

    if rads.size == 1:
        rads = jnp.ones(num_agent) * rads
    key0, key1, key2, key3 = jax.random.split(key, 4)
    start_dist = _build_target_pos("corner", "start", 0.2, 0.05)
    goal_dist = _build_target_pos("corner", "goal", 0.2, 0.05)

    carried_start = jax.random.uniform(key0, (num_agent, 2))
    start_carry = [key1, carried_start, rads, obs]
    starts = jax.lax.fori_loop(
        0,
        num_agent,
        partial(
            _sample_from_corner,
            sample_dist=start_dist,
            is_discrete=is_discrete,
            no_overlap=no_overlap,
            num_max_trials=num_max_trials,
        ),
        start_carry,
    )[1]
    carried_goal = jax.random.uniform(key2, (num_agent, 2))
    goal_carry = [key3, carried_goal, rads, obs]
    goals = jax.lax.fori_loop(
        0,
        num_agent,
        partial(
            _sample_from_corner,
            sample_dist=goal_dist,
            is_discrete=is_discrete,
            no_overlap=no_overlap,
            num_max_trials=num_max_trials,
        ),
        goal_carry,
    )[1]
    return starts, goals


def _sample_from_corner(
    i: int,
    loop_carry: list,
    sample_dist: Callable,
    is_discrete: bool,
    no_overlap: bool = False,
    num_max_trials: int = 100,
) -> list:
    """Compiled function of sample_random_pos from corner"""

    def cond(while_carry):
        (
            i,
            target_pos,
            target_rad,
            obs,
            carreid_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry[1:]
        map_size = obs.sdf.shape[0]
        pos_int = jnp.minimum((target_pos * map_size).astype(int), map_size - 1)
        if is_discrete:
            obs_overlap = jnp.array(obs.occupancy[pos_int[0], pos_int[1]], dtype=bool)
            carry_int = jnp.minimum((carried_pos * map_size).astype(int), map_size - 1)
            agent_overlap = jnp.any(jnp.all(jnp.equal(pos_int, carry_int), axis=-1))
            return jnp.any(obs_overlap, agent_overlap)
        else:
            return (
                jnp.any(
                    jnp.linalg.norm(carreid_pos - target_pos, axis=1)
                    - target_rad * 2.5
                    - rads * 2.5
                    < 0
                )
                & no_overlap
            ) | (obs.sdf[pos_int[0], pos_int[1]] - target_rad * 1.5 < 0)

    def body(while_carry):
        (
            key,
            i,
            target_pos,
            target_rad,
            obs,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ) = while_carry
        num_trials = num_trials + 1
        key0, key = jax.random.split(key)
        target_pos = sample_dist(key0, i, target_rad)
        return [
            key,
            i,
            target_pos,
            target_rad,
            obs,
            carried_pos,
            rads,
            no_overlap,
            num_trials,
        ]

    key, carried_pos, rads, obs = loop_carry
    num_trials = 0
    target_pos = carried_pos[i]
    target_rad = rads[i]

    while_carry = [
        key,
        i,
        target_pos,
        target_rad,
        obs,
        carried_pos,
        rads,
        no_overlap,
        num_trials,
    ]

    while_carry = jax.lax.while_loop(cond, body, while_carry)
    key = while_carry[0]
    pos = while_carry[2]
    num_trials = while_carry[-1]
    # pos = jax.lax.cond(
    #     num_trials < num_max_trials, lambda _: pos, lambda _: pos * jnp.inf, None
    # )
    carried_pos = carried_pos.at[i].set(pos)
    return [key, carried_pos, rads, obs]


def _build_target_pos(dist_type, pos_type, mean, sigma):
    start_mean = jnp.array(
        [[mean, mean], [mean, 1 - mean], [1 - mean, mean], [1 - mean, 1 - mean]]
    )
    goal_mean = jnp.array(
        [[1 - mean, 1 - mean], [1 - mean, mean], [mean, 1 - mean], [mean, mean]]
    )
    sigma = sigma

    def sample_target_pos(key, i, target_rad):
        if dist_type == "uniform":
            target_pos = (
                jax.random.uniform(key, shape=(2,)) * (1 - 2 * target_rad) + target_rad
            )
        else:
            if pos_type == "start":
                mean = start_mean[i % 4]
            else:
                mean = goal_mean[i % 4]

            target_pos = (
                jnp.clip(
                    jax.random.normal(key, shape=(2,)) * sigma + mean, a_max=1, a_min=0
                )
                * (1 - 2 * target_rad)
                + target_rad
            )
        return target_pos

    return sample_target_pos
