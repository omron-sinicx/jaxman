"""Definition of models used for contious SAC

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Tuple

import flax.linen as fnn
import jax.numpy as jnp
from chex import Array
from flax import linen as fnn
from jaxman.planner.rl_planner.core import AgentObservation

from .base_model import ObsActEncoder, ObsEncoder

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

epsilon = 10e-7


class Critic(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: AgentObservation,
        actions: Array,
    ) -> Array:
        """
        calculate q value

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication
            actions (Array): agent actions. shape: (batch_size, action_dim)

        Returns:
            Array: q value
        """
        # encode observation, communications and action
        encoder = ObsActEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(observations, actions)

        h = fnn.Dense(self.hidden_dim)(h)
        h = fnn.relu(h)
        q_values = fnn.Dense(1)(h)

        return q_values


class DoubleCritic(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: AgentObservation,
        actions: Array,
    ) -> Array:
        """calculate double q

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication
            actions (Array): agent action

        Returns:
            Array: double q
        """
        VmapCritic = fnn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        qs = VmapCritic(self.hidden_dim, self.msg_dim)(observations, actions)
        q1 = qs[0]
        q2 = qs[1]
        return q1, q2


class Actor(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int
    is_residual_net: bool = True
    log_std_min: float = None
    log_std_max: float = None

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Tuple[Array, Array]:
        """
        calculate agent action mean and log_std

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication

        Returns:
            Distribution: action distribution
        """
        # encode observation, communications and action
        encoder = ObsEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(observations)

        # mean
        means = fnn.Dense(self.hidden_dim)(h)
        means = fnn.relu(means)
        means = fnn.Dense(self.action_dim)(means)

        if self.is_residual_net:
            # residual network
            # apply clip to avoid inf value
            planner_act = jnp.clip(
                observations.base_observation[:, -2:],
                a_min=-1 + epsilon,
                a_max=1 - epsilon,
            )
            x_t = jnp.arctanh(planner_act)
            if self.action_dim == 2:
                means = jnp.tanh(means + x_t)
            else:
                means = means.at[:, :2].set(means[:, :2] + x_t)
                means = jnp.tanh(means)
        else:
            means = jnp.tanh(means)

        log_stds = fnn.Dense(self.hidden_dim)(h)
        log_stds = fnn.relu(log_stds)
        log_stds = fnn.Dense(self.action_dim)(log_stds)

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        return means, log_stds
