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


class Critic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
        communications: Array,
        neighbor_masks: Array,
    ) -> Array:
        """
        calculate q value

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)
            communications (Array): communications with neighbor agents. shape: (batch_size, num_comm_agents, comm_dim)
            neighbor_masks (Array): mask for obtaining only neighboring agent communications. shape: (batch_size, num_agents)

        Returns:
            Array: q value
        """
        # encode observation, communications and action
        encoder = ObsEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(observations, communications, neighbor_masks)

        h = fnn.Dense(self.hidden_dim)(h)
        h = fnn.relu(h)
        q_values = fnn.Dense(self.action_dim)(h)

        return q_values


class DoubleCritic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(
        self, observations: Array, communications: Array, neighbor_masks: Array
    ) -> Array:
        """calculate double q

        Args:
            observations (Array): agent observation
            communications (Array): communications with neighbor agents.
            neighbor_masks (Array): mask for obtaining only neighboring agent communications. shape: (batch_size, num_agents)

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
            communications,
            neighbor_masks,
        )
        q1 = qs[0]
        q2 = qs[1]
        return q1, q2


class Actor(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
        communications: Array,
        neighbor_masks: Array,
    ) -> Array:
        """
        calculate agent action distribution

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)
            communications (Array): communications with neighbor agents. shape: (batch_size, num_comm_agents, comm_dim)
            neighbor_masks (Array): mask for obtaining only neighboring agent communications. shape: (batch_size, num_agents)
        Returns:
            Array: action distribution
        """
        # encode observation, communications and action
        encoder = ObsEncoder(self.hidden_dim, self.msg_dim)
        h = encoder(
            observations,
            communications,
            neighbor_masks,
        )

        action_logits = fnn.Dense(self.hidden_dim)(h)
        action_logits = fnn.relu(action_logits)
        action_logits = fnn.Dense(self.action_dim)(action_logits)
        action_probs = fnn.softmax(action_logits, axis=-1)

        base_dist = tfd.Categorical(probs=action_probs)
        return base_dist
