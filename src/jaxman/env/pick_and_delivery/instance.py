"""Definition of problem instance

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from omegaconf.dictconfig import DictConfig

from ..core import AgentInfo, EnvInfo
from ..obstacle import ObstacleMap
from ..task_generator import generate_obs_map, sample_valid_agent_item_pos
from .core import TaskInfo


# @dataclass
class Instance:
    """Environment instance of multi-agent pick and delivery"""

    env_name: str
    num_agents: int
    num_items: int
    max_vels: Array
    max_ang_vels: Array
    rads: Array
    item_rads: Array
    goal_rads: Array
    obs: ObstacleMap
    fov_r: int
    comm_r: float
    num_scans: int
    scan_range: float
    use_hold_item_info: bool
    timeout: int
    is_crashable: bool
    is_biased_sample: bool
    is_respawn: bool
    goal_reward: float
    dist_reward: float
    dont_hold_item_penalty: float
    crash_penalty: float
    time_penalty: float
    pickup_reward: float
    life_penalty: float
    is_decay_reward: bool
    decay_start: int
    decay_end: int
    min_reward: float
    agent_starts: Array
    agent_start_rots: Array
    item_starts: Array
    item_goals: Array
    goals: Array
    is_discrete: bool
    is_diff_drive: bool

    def __init__(self, config: DictConfig) -> None:
        self.env_name = config.env_name
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.map_size = config.map_size
        self.max_vels = jnp.array([[config.max_vel] for _ in range(self.num_agents)])
        self.min_vels = jnp.array([[config.min_vel] for _ in range(self.num_agents)])
        self.max_ang_vels = jnp.array(
            [[config.max_ang_vel * jnp.pi] for _ in range(self.num_agents)]
        )
        self.min_ang_vels = jnp.array(
            [[config.min_ang_vel * jnp.pi] for _ in range(self.num_agents)]
        )
        self.max_accs = jnp.array([[config.max_acc] for _ in range(self.num_agents)])
        self.max_ang_accs = jnp.array(
            [[config.max_ang_acc * jnp.pi] for _ in range(self.num_agents)]
        )
        self.rads = jnp.array([[config.rad] for _ in range(self.num_agents)])
        self.item_rads = jnp.array([[config.rad] for _ in range(self.num_items)])
        self.goal_rads = jnp.array([[config.goal_rad] for _ in range(self.num_agents)])
        self.fov_r = config.fov_r
        self.comm_r = config.comm_r
        self.num_scans = config.num_scans
        self.scan_range = config.scan_range
        self.use_intentions = config.use_intentions
        self.use_hold_item_info = config.use_hold_item_info
        self.timeout = config.timeout
        self.is_crashable = config.is_crashable
        self.is_biased_sample = config.is_biased_sample
        self.is_respawn = config.is_respawn
        self.goal_reward = config.goal_reward
        self.dist_reward = config.dist_reward
        self.dont_hold_item_penalty = config.dont_hold_item_penalty
        self.crash_penalty = config.crash_penalty
        self.time_penalty = config.time_penalty
        self.pickup_reward = config.pickup_reward
        self.life_penalty = config.life_penalty
        self.is_decay_reward = config.is_decay_reward
        self.decay_start = config.decay_start
        self.decay_end = config.decay_end
        self.min_reward = config.min_reward

        self.is_discrete = config.is_discrete
        self.is_diff_drive = config.is_diff_drive

        self.obs = generate_obs_map(
            config.obs_type,
            config.level,
            config.is_discrete,
            config.map_size,
            config.obs_size_lower_bound,
            config.obs_size_upper_bound,
            jax.random.PRNGKey(46),
        )

        key_start_goal = jax.random.split(jax.random.PRNGKey(46))
        (
            agent_starts,
            agent_start_rots,
            item_starts,
            item_goals,
        ) = sample_valid_agent_item_pos(
            key_start_goal,
            self.rads,
            self.obs,
            self.num_agents,
            self.num_items,
            config.is_discrete,
            config.is_diff_drive,
            is_biased_sample=self.is_biased_sample,
        )
        self.agent_starts = agent_starts
        self.agent_start_rots = agent_start_rots
        self.item_starts = item_starts
        self.item_goals = item_goals

    def export_info(
        self,
    ) -> tuple[EnvInfo, AgentInfo, TaskInfo]:
        """
        Export a set of information used for local planning.

        Returns:
            tuple[EnvInfo, AgentInfo, TaskInfo]: a set of information
        """

        env_info = EnvInfo(
            env_name="pick_and_delivery",
            num_agents=int(self.num_agents),
            num_items=self.num_items,
            occupancy_map=self.obs.occupancy,
            sdf_map=self.obs.sdf,
            edges=self.obs.edges,
            item_rads=self.item_rads,
            fov_r=int(self.fov_r),
            comm_r=self.comm_r,
            num_scans=int(self.num_scans),
            scan_range=float(self.scan_range),
            use_intentions=bool(self.use_intentions),
            use_hold_item_info=bool(self.use_hold_item_info),
            timeout=int(self.timeout),
            is_crashable=bool(self.is_crashable),
            is_biased_sample=bool(self.is_biased_sample),
            is_respawn=bool(self.is_respawn),
            goal_reward=self.goal_reward,
            dist_reward=self.dist_reward,
            dont_hold_item_penalty=self.dont_hold_item_penalty,
            crash_penalty=self.crash_penalty,
            time_penalty=self.time_penalty,
            pickup_reward=self.pickup_reward,
            life_penalty=self.life_penalty,
            is_decay_reward=self.is_decay_reward,
            decay_start=self.decay_start,
            decay_end=self.decay_end,
            min_reward=self.min_reward,
            is_discrete=self.is_discrete,
            is_diff_drive=self.is_diff_drive,
        )
        agent_info = AgentInfo(
            max_vels=self.max_vels,
            min_vels=self.min_vels,
            max_ang_vels=self.max_ang_vels,
            min_ang_vels=self.min_ang_vels,
            max_accs=self.max_accs,
            max_ang_accs=self.max_ang_accs,
            rads=self.rads,
        )
        task_info = TaskInfo(
            starts=self.agent_starts,
            start_rots=self.agent_start_rots,
            item_starts=self.item_starts,
            item_goals=self.item_goals,
            obs=self.obs,
        )

        return env_info, agent_info, task_info

    def reset(self, key: PRNGKey) -> TaskInfo:
        """reset task information (reset start position, rotation and goal position)

        Args:
            key (PRNGKey): random key variable

        Returns:
            TaskInfo: reset task information
        """
        (
            agent_starts,
            agent_start_rots,
            item_starts,
            item_goals,
        ) = sample_valid_agent_item_pos(
            key,
            self.rads,
            self.obs,
            self.num_agents,
            self.num_items,
            self.is_discrete,
            self.is_diff_drive,
            is_biased_sample=self.is_biased_sample,
        )
        self.agent_starts = agent_starts
        self.agent_start_rots = agent_start_rots
        self.item_starts = item_starts
        self.item_goals = item_goals
        return TaskInfo(
            starts=self.agent_starts,
            start_rots=self.agent_start_rots,
            item_starts=self.item_starts,
            item_goals=self.item_goals,
            obs=self.obs,
        )

    def reset_obs(
        self,
        obs_type: str,
        level: int,
        map_size: int,
        obs_size_lower_bound: float,
        obs_size_upper_bound: float,
        key: PRNGKey,
    ) -> TaskInfo:
        """reset task information including obstacle placement

        Args:
            obs_type (str): obstacle placement type
            level (int): obstacle placement level
            map_size (int): map size
            obs_size_lower_bound (float): lower bound of obstacle radius
            obs_size_upper_bound (float): upper bound of obstacle radius
            key (PRNGKey): random key variable

        Returns:
            TaskInfo: reset task information
        """
        self.obs = generate_obs_map(
            obs_type,
            level,
            self.is_discrete,
            map_size,
            obs_size_lower_bound,
            obs_size_upper_bound,
            key,
        )
        key_start_goal = jax.random.split(key)
        (
            agent_starts,
            agent_start_rots,
            item_starts,
            item_goals,
        ) = sample_valid_agent_item_pos(
            key_start_goal,
            self.rads,
            self.obs,
            self.num_agents,
            self.num_items,
            self.is_discrete,
            self.is_diff_drive,
        )
        self.agent_starts = agent_starts
        self.agent_start_rots = agent_start_rots
        self.item_starts = item_starts
        self.item_goals = item_goals
        return TaskInfo(
            starts=self.agent_starts,
            start_rots=self.agent_start_rots,
            item_starts=self.item_starts,
            item_goals=self.item_goals,
            obs=self.obs,
        )
