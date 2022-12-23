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
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax.distributions import Distribution

from .base_model import ObsActEncoder, ObsEncoder

tfd = tfp.distributions
tfb = tfp.bijectors
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
    is_residual_net: bool = False
    log_std_min: float = None
    log_std_max: float = None

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Tuple[Array, Array]:
        """
        calculate agent action distribution

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
                observations[:, -2:], a_min=-1 + epsilon, a_max=1 - epsilon
            )
            x_t = jnp.arctanh(planner_act)
            means = jnp.tanh(means + x_t)
        else:
            means = jnp.tanh(means)

        log_stds = fnn.Dense(self.hidden_dim)(h)
        log_stds = fnn.relu(log_stds)
        log_stds = fnn.Dense(self.action_dim)(log_stds)

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        # base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return means, log_stds
        # return base_dist
