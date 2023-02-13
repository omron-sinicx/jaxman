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
from ..task_generator import generate_obs_map, sample_valid_start_goal
from .core import TaskInfo


# @dataclass
class Instance:
    """Problem instance of multi-agent navigation"""

    env_name: str
    num_agents: int
    max_vels: Array
    max_ang_vels: Array
    rads: Array
    goal_rads: Array
    obs: ObstacleMap
    fov_r: int
    comm_r: float
    num_scans: int
    scan_range: float
    timeout: int
    goal_reward: float
    crash_penalty: float
    time_penalty: float
    starts: Array
    start_rots: Array
    goals: Array
    is_discrete: bool
    is_diff_drive: bool

    def __init__(self, config: DictConfig) -> None:
        self.env_name = config.env_name
        self.num_agents = config.num_agents
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
        self.goal_rads = jnp.array([[config.goal_rad] for _ in range(self.num_agents)])
        self.fov_r = config.fov_r
        self.comm_r = config.comm_r
        self.num_scans = config.num_scans
        self.scan_range = config.scan_range
        self.use_intentions = config.use_intentions
        self.timeout = config.timeout
        self.goal_reward = config.goal_reward
        self.crash_penalty = config.crash_penalty
        self.time_penalty = config.time_penalty

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
        starts, start_rots, goals = sample_valid_start_goal(
            key_start_goal,
            self.rads,
            self.obs,
            self.num_agents,
            config.is_discrete,
            config.is_diff_drive,
        )
        self.starts = starts
        self.start_rots = start_rots
        self.goals = goals

    def export_info(
        self,
    ) -> tuple[EnvInfo, AgentInfo, TaskInfo]:
        """
        Export a set of information used for local planning.

        Returns:
            tuple[EnvInfo, AgentInfo, TaskInfo]: a set of information
        """

        env_info = EnvInfo(
            env_name="navigation",
            num_agents=int(self.num_agents),
            num_items=0,
            occupancy_map=self.obs.occupancy,
            sdf_map=self.obs.sdf,
            edges=self.obs.edges,
            fov_r=int(self.fov_r),
            comm_r=self.comm_r,
            num_scans=int(self.num_scans),
            scan_range=float(self.scan_range),
            use_intentions=bool(self.use_intentions),
            timeout=int(self.timeout),
            is_crashable=True,
            goal_reward=self.goal_reward,
            dist_reward=0,
            dont_hold_item_penalty=0,
            crash_penalty=self.crash_penalty,
            time_penalty=self.time_penalty,
            pickup_reward=0,
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
            starts=self.starts,
            start_rots=self.start_rots,
            goals=self.goals,
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
        key_start_goal = jax.random.split(key)
        starts, start_rots, goals = sample_valid_start_goal(
            key_start_goal,
            self.rads,
            self.obs,
            self.num_agents,
            self.is_discrete,
            self.is_discrete,
        )
        self.starts = starts
        self.start_rots = start_rots
        self.goals = goals
        return TaskInfo(
            starts=self.starts,
            start_rots=self.start_rots,
            goals=self.goals,
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
        starts, start_rots, goals = sample_valid_start_goal(
            key_start_goal, self.rads, self.obs, self.num_agents
        )
        self.starts = starts
        self.start_rots = start_rots
        self.goals = goals
        return TaskInfo(
            starts=self.starts,
            start_rots=self.start_rots,
            goals=self.goals,
            obs=self.obs,
        )
