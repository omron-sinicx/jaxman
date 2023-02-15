"""Jax-based environment for multi-agent pick and delivery problem

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
from ..pick_and_delivery.instance import Instance
from ..viz.viz import render_gif, render_map
from .core import State, TaskInfo, TrialInfo
from .dynamics import _build_inner_step
from .observe import _build_observe


class JaxPandDEnv(gym.Env):
    def __init__(self, config: DictConfig, seed: int = 0):
        super().__init__()
        self.key = jax.random.PRNGKey(seed)
        self.instance = Instance(config)
        self._env_info, self._agent_info, self._task_info = self.instance.export_info()
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.is_discrete = config.is_discrete
        self.is_diff_drive = config.is_diff_drive
        self._step = _build_inner_step(self._env_info, self._agent_info)
        self._observe = _build_observe(self._env_info, self._agent_info)
        self.obs = self.reset()
        self._create_spaces(config.is_discrete)

    def _create_spaces(self, is_discrete: bool):
        if is_discrete:
            self.act_space = Discrete(6)
        else:
            self.act_space = Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        if not self.is_discrete:
            comm_dim = 10  # (rel_pos, rot, vel, ang) * 2
        elif self.is_diff_drive:
            comm_dim = 6  # (rel_pos, rot) * 2
        else:
            comm_dim = 4  # (rel_pos,) * 2
        item_dim = 4  # (item_pos, item_goal)

        obs_dim = (
            self.obs.cat()[0].shape[0]
            - comm_dim * self.num_agents
            - self.num_agents
            - item_dim * self.num_items
            - self.num_items
        )
        obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        comm_space = Box(
            low=-1.0, high=1.0, shape=(self.num_agents, comm_dim), dtype=np.float32
        )
        mask_space = Box(low=0.0, high=1.0, shape=(self.num_agents,))
        item_space = Box(low=0.0, high=1.0, shape=(self.num_items, item_dim))
        item_mask_space = Box(low=0.0, high=1.0, shape=(self.num_items,))
        self.obs_space = Dict(
            {
                "obs": obs_space,
                "comm": comm_space,
                "mask": mask_space,
                "item_pos": item_space,
                "item_mask": item_mask_space,
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
        agent_state = AgentState(
            pos=self.task_info.starts,
            rot=self.task_info.start_rots,
            vel=jnp.zeros_like(self.task_info.start_rots, dtype=int),
            ang=jnp.zeros_like(self.task_info.start_rots, dtype=int),
        )
        self.state = State(
            agent_state,
            jnp.ones((self.num_agents,), dtype=int) * self.num_items,
            self.task_info.item_starts,
        )

        # initialize trial information
        self.trial_info = TrialInfo().reset(self.num_agents, self.num_items)
        return self._observe(self.state, self.task_info, self.trial_info)

    def render(self, state_traj=None, dones=None, high_quality=False) -> np.Array:
        return render_map(
            self.state,
            self.task_info.item_goals,
            self._agent_info.rads,
            self.task_info.obs.occupancy,
            state_traj,
            self.trial_info,
            dones,
            self.is_discrete,
            self.is_diff_drive,
            high_quality,
            "pick_and_delivery",
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

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        """
        Receives all agent actions as an array
        Returns the observation, reward, terminated, truncated and info
        """
        state, rew, done, info = self._step(
            self.state, actions, self.task_info, self.trial_info
        )
        self.state = state
        self.trial_info = info
        obs = self._observe(state, self.task_info, self.trial_info)
        return obs, rew, done, info
