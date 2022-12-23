"""Definition of models used for discrete SAC

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import flax.linen as fnn
from chex import Array
from flax import linen as fnn
from tensorflow_probability.substrates import jax as tfp

from .base_model import ObsEncoder

tfd = tfp.distributions
tfb = tfp.bijectors
from jaxman.planner.rl_planner.core import AgentObservation


class Critic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Array:
        """
        calculate q value

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication

        Returns:
            Array: q value
        """
        # encode observation, communications and action
        encoder = ObsEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(observations)

        h = fnn.Dense(self.hidden_dim)(h)
        h = fnn.relu(h)
        q_values = fnn.Dense(self.action_dim)(h)

        return q_values


class DoubleCritic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Array:
        """calculate double q

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication

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
        qs = VmapCritic(self.hidden_dim, self.msg_dim, self.action_dim)(
            observations,
        )
        q1 = qs[0]
        q2 = qs[1]
        return q1, q2


class Actor(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Array:
        """
        calculate agent action distribution

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication
        Returns:
            Array: action distribution
        """
        # encode observation, communications and action
        encoder = ObsEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(
            observations,
        )

        action_logits = fnn.Dense(self.hidden_dim)(h)
        action_logits = fnn.relu(action_logits)
        action_logits = fnn.Dense(self.action_dim)(action_logits)
        action_probs = fnn.softmax(action_logits, axis=-1)

        # base_dist = tfd.Categorical(probs=action_probs)
        return action_probs
