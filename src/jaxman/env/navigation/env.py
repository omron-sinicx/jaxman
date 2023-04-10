"""Jax-based environment for multi-agent navigation problem

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from __future__ import annotations

import gym
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pylab as plt
import numpy as np
from chex import Array, PRNGKey
from gym.spaces import Box, Dict, Discrete
from omegaconf.dictconfig import DictConfig

from ..core import AgentState
from ..viz.viz import render_gif, render_map
from .core import TaskInfo, TrialInfo
from .dynamics import _build_inner_step
from .instance import Instance
from .observe import _build_observe


class JaxMANEnv(gym.Env):
    def __init__(self, config: DictConfig, seed: int = 0):
        super().__init__()
        self.key = jax.random.PRNGKey(seed)
        self.instance = Instance(config)
        self._env_info, self._agent_info, self._task_info = self.instance.export_info()
        self.num_agents = config.num_agents
        self.is_discrete = config.is_discrete
        self.is_diff_drive = config.is_diff_drive
        self.use_intentions = config.use_intentions
        self._inner_step = _build_inner_step(self._env_info, self._agent_info)
        self._observe = _build_observe(self._env_info, self._agent_info)
        self.obs = self.reset()
        self._create_spaces(config.is_discrete)

    def _create_spaces(self, is_discrete: bool):
        if is_discrete:
            self.act_space = Discrete(5)
        else:
            self.act_space = Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        if not self.is_discrete:
            comm_dim = 5  # (rel_pos, rot, vel, ang) * 2
        elif self.is_diff_drive:
            comm_dim = 3  # (rel_pos, rot) * 2
        else:
            comm_dim = 2  # (rel_pos,) * 2
        if self.use_intentions:
            comm_dim *= 2

        obs_dim = (
            self.obs.cat()[0].shape[0] - comm_dim * self.num_agents - self.num_agents
        )
        obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        comm_space = Box(
            low=-1.0, high=1.0, shape=(self.num_agents, comm_dim), dtype=np.float32
        )
        mask_space = Box(low=0.0, high=1.0, shape=(self.num_agents,))
        item_pos_shape = item_mask_shape = Box(low=0.0, high=0.0, shape=(0,))
        self.obs_space = Dict(
            {
                "obs": obs_space,
                "comm": comm_space,
                "mask": mask_space,
                "item_pos": item_pos_shape,
                "item_mask": item_mask_shape,
            }
        )

    @property
    def observation_space(self) -> gym.Space:
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        return self.act_space

    def sample_actions(self) -> Array:
        return jnp.array([self.action_space.sample() for _ in range(self.num_agents)])

    def reset(self, key: PRNGKey = None, task_info: TaskInfo = None) -> dict:
        # initialize task_info and dynamics accordingly
        if task_info is not None:
            self.task_info = task_info

        else:
            if key is not None:
                new_key = key
            else:
                new_key, self.key = jax.random.split(self.key)
            self.task_info = self.instance.reset(
                new_key,
            )

        # initialize state
        self.state = AgentState(
            pos=self.task_info.starts,
            rot=self.task_info.start_rots,
            vel=jnp.zeros_like(self.task_info.start_rots),
            ang=jnp.zeros_like(self.task_info.start_rots),
        )

        # initialize trial information
        self.trial_info = TrialInfo().reset(self._env_info.num_agents)
        return self._observe(
            self.state, self.task_info, jnp.ones((self.num_agents,), dtype=bool)
        )

    def render(self, state_traj=None, dones=None, high_quality=False) -> np.Array:
        return render_map(
            self.state,
            self.task_info.goals,
            self._agent_info.rads,
            self.task_info.obs.occupancy,
            state_traj,
            self.trial_info,
            dones,
            self.is_discrete,
            self.is_diff_drive,
            high_quality,
        )

    def save_gif(self, state_traj, dones, high_quality=True, filename="result.gif"):
        render_image = render_gif(
            state_traj,
            self.task_info.goals,
            self._agent_info.rads,
            self.task_info.obs.occupancy,
            self.trial_info,
            dones,
            self.is_discrete,
            self.is_diff_drive,
            high_quality,
        )
        fig, ax = plt.subplots()

        # Adjust figure so GIF does not have extra whitespace
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis("off")

        imgs = []
        for i in range(len(render_image)):
            im = ax.imshow(render_image[i])
            imgs.append([im])

        ani = animation.ArtistAnimation(fig, imgs, interval=600)
        ani.save(filename)

    def state(self) -> Array:
        return self.state

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        """
        Receives all agent actions as an array
        Returns the observation, reward, terminated, truncated and info
        """
        state, rew, done, info = self._inner_step(
            self.state, actions, self.task_info, self.trial_info
        )
        self.state = state
        self.trial_info = info
        obs = self._observe(state, self.task_info, jnp.logical_not(done))
        return obs, rew, done, info

    # def build_step(self, is_discrete: bool, is_diff_drive: bool):
    #     return _build_step(self._env_info, self._agent_info, is_discrete, is_diff_drive)

    def build_reset(self):
        pass
