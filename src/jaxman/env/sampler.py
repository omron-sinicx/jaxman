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


@partial(jax.jit, static_argnames=("num_samples", "is_discrete"))
def sample_start_rots(key: PRNGKey, num_samples: int, is_discrete: bool) -> Array:
    """sample agent start rotations

    Args:
        key (PRNGKey): random key variable
        num_samples (int): number to sample
        is_discrete (bool): whether environment is discrete space or not

    Returns:
        Array: sample agent start rotations
    """
    if is_discrete:
        start_rots = jnp.zeros((num_samples, 1), dtype=int)
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
    start_key, goal_key = jax.random.split(key)
    if sample_type == "uniform":
        starts = sample_uniform(
            start_key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            no_overlap,
            num_max_trials,
        )
        goals = sample_uniform(
            goal_key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            no_overlap,
            num_max_trials,
        )
    else:
        starts = sample_from_corner(
            start_key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            "start",
            no_overlap,
            num_max_trials,
        )
        goals = sample_from_corner(
            goal_key,
            num_agents,
            rads.flatten(),
            obs,
            is_discrete,
            "goals",
            no_overlap,
            num_max_trials,
        )
    if is_discrete:
        map_size = obs.sdf.shape[0]
        starts = jnp.minimum((starts * map_size).astype(int), map_size - 1)
        goals = jnp.minimum((goals * map_size).astype(int), map_size - 1)
    return starts, goals


@partial(
    jax.jit,
    static_argnames=(
        "num_agents",
        "num_items",
        "num_max_trials",
        "is_discrete",
        "sample_type",
        "is_biased_sample",
    ),
)
def sample_random_agent_item_pos(
    key: PRNGKey,
    rads: Array,
    obs: ObstacleMap,
    num_agents: int,
    num_items: int,
    is_discrete: bool,
    no_overlap: bool = True,
    sample_type: str = "uniform",
    is_biased_sample: bool = False,
    num_max_trials: bool = 200,
) -> Tuple[Array, Array]:
    """sample agent start and goal position

    Args:
        key (PRNGKey): random key variables
        rads (Array): agent radius
        obs (ObstacleMap): obstacle map
        num_agents (int): number of agent
        num_items (int): number of items in environment
        is_discrete (bool): whether environment is discrete space or not
        no_overlap (bool, optional): whether to allow overlap in sampled positions. Defaults to True.
        sample_type (str, optional): sample type. Defaults to "uniform".
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Tuple[Array, Array]: sampled agent start and goal
    """
    agent_key, item_key = jax.random.split(key)
    max_rad = jnp.max(rads.flatten())
    # if sample_type == "uniform":
    agent_starts = sample_uniform(
        agent_key,
        num_agents,
        max_rad,
        obs,
        is_discrete,
        no_overlap,
        num_max_trials,
    )
    # else:
    #     agent_item_starts_goals = sample_from_corner(
    #         start_key,
    #         num_agents + num_items * 2,
    #         max_rad,
    #         obs,
    #         is_discrete,
    #         "start",
    #         no_overlap,
    #         num_max_trials,
    #     )
    if is_discrete:
        map_size = obs.sdf.shape[0]
        agent_starts = jnp.minimum((agent_starts * map_size).astype(int), map_size - 1)
        item_starts, item_goals = _sample_item(
            item_key, agent_starts, obs.occupancy, num_items, map_size, is_biased_sample
        )
    else:
        item_pos = sample_uniform(
            item_key,
            num_items * 2,
            max_rad,
            obs,
            is_discrete,
            no_overlap,
            num_max_trials,
        )
        item_starts = item_pos[:num_items]
        item_goals = item_pos[num_items:]
    return agent_starts, item_starts, item_goals


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
    key0, key1 = jax.random.split(key)
    carried_pos = jax.random.uniform(key0, (num_samples, 2))

    loop_carry = [key1, carried_pos, rads, obs]
    pos = jax.lax.fori_loop(
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
    return pos


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
            # Ensure that there are no obstacles above or below the currently sampled position
            obs_overlap = jnp.any(
                jax.lax.dynamic_slice_in_dim(
                    obs.occupancy[pos_int[0]], pos_int[1] - 1, 3, 0
                )
            )
            pos = jnp.minimum((carried_pos * map_size).astype(int), map_size - 1)
            pos_with_above = jnp.concatenate((pos, pos + jnp.array([0, 1])))
            pos_with_above_below = jnp.concatenate(
                (pos_with_above, pos + jnp.array([0, -1]))
            )
            agent_overlap = jnp.any(
                jnp.all(jnp.equal(pos_int, pos_with_above_below), axis=-1)
            )
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


@partial(
    jax.jit, static_argnames=("num_agent", "num_max_trials", "is_discrete", "pos_type")
)
def sample_from_corner(
    key: PRNGKey,
    num_agent: int,
    rads: Array,
    obs: ObstacleMap,
    is_discrete: bool,
    pos_type: str,
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
        pos_type (str): type of position to sample (start or goal)
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        Array: list of random valid positions
    """

    if rads.size == 1:
        rads = jnp.ones(num_agent) * rads
    key0, key1 = jax.random.split(key, 4)

    sample_dist = _build_target_pos("corner", pos_type, 0.2, 0.05)

    carried_start = jax.random.uniform(key0, (num_agent, 2))
    start_carry = [key1, carried_start, rads, obs]
    pos = jax.lax.fori_loop(
        0,
        num_agent,
        partial(
            _sample_from_corner,
            sample_dist=sample_dist,
            is_discrete=is_discrete,
            no_overlap=no_overlap,
            num_max_trials=num_max_trials,
        ),
        start_carry,
    )[1]
    return pos


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


@partial(jax.jit, static_argnames=("num_items", "map_size", "is_biased_sample"))
def _sample_item(
    key: PRNGKey,
    agent_pos: Array,
    obs_map: Array,
    num_items: int,
    map_size: int,
    is_biased_sample: bool,
):
    def _index_to_xy(item_pos, map_size):
        return jnp.array([(item_pos / map_size).astype(int), item_pos % map_size])

    agent_item_obs_map = jnp.sum(
        jax.vmap(
            lambda pos, map: map.at[pos[0], pos[1]].set(1.0),
            in_axes=(0, None),
        )(agent_pos, obs_map),
        axis=0,
    )
    b = agent_item_obs_map[:, 1:] + agent_item_obs_map[:, :-1]
    agent_item_obs_map = agent_item_obs_map.at[:, :-1].set(b).astype(bool).astype(int)
    sample_prob = 1 - agent_item_obs_map.reshape(-1)
    key, subkey = jax.random.split(key)
    if is_biased_sample:
        biased_mask = (jnp.arange(map_size**2) < (map_size**2 / 2)).astype(bool)

        start_prob = sample_prob + sample_prob * biased_mask * 10
        start_prob = start_prob / jnp.sum(start_prob)
        item_start = jax.random.choice(
            subkey, map_size**2, shape=(num_items,), replace=False, p=start_prob
        )

        goal_prob = sample_prob + sample_prob * (1 - biased_mask) * 10
        goal_prob = goal_prob / jnp.sum(goal_prob)
        item_goal = jax.random.choice(
            subkey, map_size**2, shape=(num_items,), replace=False, p=goal_prob
        )
    else:
        subkey1, subkey2 = jax.random.split(subkey)
        sample_prob = sample_prob / jnp.sum(sample_prob)
        item_start = jax.random.choice(
            subkey1, map_size**2, shape=(num_items,), replace=False, p=sample_prob
        )
        item_goal = jax.random.choice(
            subkey2, map_size**2, shape=(num_items,), replace=False, p=sample_prob
        )

    item_start = jax.vmap(_index_to_xy, in_axes=[0, None])(item_start, map_size)
    item_goal = jax.vmap(_index_to_xy, in_axes=[0, None])(item_goal, map_size)
    return item_start, item_goal
