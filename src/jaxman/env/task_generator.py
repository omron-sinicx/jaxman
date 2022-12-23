"""multi agent navigation problem generator. generate initial start and goal positions, obstacle map

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from functools import partial

import jax
import numpy as np
from chex import Array, PRNGKey
from CMap2D import CMap2D

from .core import AgentState
from .obstacle import ObstacleMap, ObstacleSphere, contours_to_edges, cross_road, room
from .sampler import sample_random_agent_item_pos, sample_random_pos, sample_start_rots


@partial(
    jax.jit,
    static_argnames=("num_agents", "sample_type", "is_discrete", "is_diff_drive"),
)
def sample_valid_start_goal(
    key: PRNGKey,
    rads: Array,
    obs: ObstacleMap,
    num_agents: int,
    is_discrete: bool,
    is_diff_drive: bool,
    no_overlap: bool = True,
    sample_type: str = "uniform",
    num_max_trials: int = 200,
) -> tuple[Array, Array, Array]:
    """
    Sample new problem instance consisting of a collection of start poses and goal positions

    Args:
        key (PRNGKey): jax prng key
        rads (Array): agent radius
        obs (ObstacleMap): obstacle map
        num_agents (int): number of agents
        is_discrete (bool): whether environment action space is discrete or not
        is_diff_drive (bool): whether environment is diff drive env or not
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        tuple[AgentState, Array, PRNGKey]: start position, start rotation, goal position
    """

    key_pos, key_pose = jax.random.split(key, 2)
    starts, goals = sample_random_pos(
        key_pos,
        rads,
        obs,
        num_agents,
        is_discrete,
        no_overlap,
        sample_type,
        num_max_trials,
    )
    start_rots = sample_start_rots(key_pose, num_agents, is_discrete, is_diff_drive)

    return starts, start_rots, goals


@partial(
    jax.jit,
    static_argnames=(
        "num_agents",
        "num_items",
        "sample_type",
        "is_discrete",
        "is_diff_drive",
    ),
)
def sample_valid_agent_item_pos(
    key: PRNGKey,
    rads: Array,
    obs: ObstacleMap,
    num_agents: int,
    num_items: int,
    is_discrete: bool,
    is_diff_drive: bool,
    no_overlap: bool = True,
    sample_type: str = "uniform",
    num_max_trials: int = 200,
) -> tuple[Array, Array, Array]:
    """
    Sample new problem instance consisting of a collection of start poses and goal positions

    Args:
        key (PRNGKey): jax prng key
        rads (Array): agent radius
        obs (ObstacleMap): obstacle map
        num_agents (int): number of agents
        is_discrete (bool): whether environment action space is discrete or not
        is_diff_drive (bool): whether environment is diff drive env or not
        no_overlap (bool, optional): whether or not to allow each vertices to be overlapped within agent radius. Defaults to False.
        num_max_trials (int, optional): maximum number of resampling. Defaults to 100.

    Returns:
        tuple[AgentState, Array, PRNGKey]: start position, start rotation, goal position
    """

    key_pos, key_pose = jax.random.split(key, 2)
    agent_starts, item_starts, item_goals = sample_random_agent_item_pos(
        key_pos,
        rads,
        obs,
        num_agents,
        num_items,
        is_discrete,
        no_overlap,
        sample_type,
        num_max_trials,
    )
    agent_start_rots = sample_start_rots(
        key_pose, num_agents, is_discrete, is_diff_drive
    )

    return agent_starts, agent_start_rots, item_starts, item_goals


# obstacle map generator
def generate_obs_map(
    obs_type: str,
    level: int,
    is_discrete: bool,
    map_size: int = 128,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.08,
    key: PRNGKey = jax.random.PRNGKey(46),
) -> ObstacleMap:
    """generate obstacle map

    Args:
        obs_type (str): obstacle map type
        level (int): level of obstacle placement
        is_discrete (bool): whether environment action space is discrete or not
        map_size (int, optional): map size. Defaults to 128.
        obs_size_lower_bound (float, optional): lower bound of obstacle radius. Defaults to 0.05.
        obs_size_upper_bound (float, optional): upper bound of obstacle radius. Defaults to 0.08.
        key (PRNGKey, optional): random key variable . Defaults to jax.random.PRNGKey(46).

    Returns:
        ObstacleMap: obstacle map
    """
    if obs_type == "random":
        occupancy = generate_random_occupancy_map(
            level,
            is_discrete,
            map_size,
            obs_size_lower_bound,
            obs_size_upper_bound,
            key,
        )
    else:
        occupancy = generata_image_base_occupancy_map(obs_type, level, map_size)
    occupancy[0, :] = 1
    occupancy[-1, :] = 1
    occupancy[:, 0] = 1
    occupancy[:, -1] = 1

    cmap2d = CMap2D()
    cmap2d.from_array(occupancy, (0, 0), 1.0 / map_size)
    sdf = cmap2d.as_sdf()
    edges = contours_to_edges(cmap2d.as_closed_obst_vertices())
    obs = ObstacleMap(occupancy, sdf, edges)
    return obs


def generate_random_occupancy_map(
    num_obs: int,
    is_discrete: bool,
    map_size: int = 128,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.08,
    key: PRNGKey = jax.random.PRNGKey(46),
) -> Array:
    """generate obstacle occupancy map

    Args:
        num_obs (int): _description_
        is_discrete (bool): _description_
        map_size (int, optional): _description_. Defaults to 128.
        obs_size_lower_bound (float, optional): _description_. Defaults to 0.05.
        obs_size_upper_bound (float, optional): _description_. Defaults to 0.08.
        key (PRNGKey, optional): _description_. Defaults to jax.random.PRNGKey(46).

    Returns:
        Array: _description_
    """
    if is_discrete:
        occupancy = np.zeros((map_size, map_size))
        obs = jax.random.uniform(key, shape=(num_obs, 2))
        for i in range(len(obs)):
            occupancy[int(obs[i, 0] * map_size), int(obs[i, 1] * map_size)] = 1
    else:
        circle_obs = jax.random.uniform(key, shape=(num_obs, 3))
        circle_obs = circle_obs.at[:, 2].multiply(
            (obs_size_upper_bound - obs_size_lower_bound) / 2
        )
        circle_obs = circle_obs.at[:, 2].add(obs_size_lower_bound / 2)
        circle_obs = [ObstacleSphere(pos=o[:2], rad=o[2]) for o in circle_obs]
        occupancy = np.dstack([x.draw_2d(map_size) for x in circle_obs]).max(-1)
    return occupancy


def generata_image_base_occupancy_map(obs_type: str, level: int, map_size: int = 128):
    if obs_type == "cross_road":
        return cross_road(map_size, level)
    elif obs_type == "room":
        return room(map_size, level)
