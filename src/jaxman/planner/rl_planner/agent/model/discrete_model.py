"""Definition of models used for discrete SAC

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import flax.linen as fnn
from chex import Array, assert_shape
from flax import linen as fnn
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class Critic(fnn.Module):
    hidden_dim: int
    action_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
    ) -> Array:
        """
        calculate q value

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)

        Returns:
            Array: q value
        """
        batch_size = observations.shape[0]

        h_obs = fnn.Dense(self.hidden_dim)(observations)
        h_obs = fnn.relu(h_obs)
        h_obs = fnn.Dense(self.hidden_dim)(h_obs)
        h_obs = fnn.relu(h_obs)

        q_values = fnn.Dense(self.hidden_dim)(h_obs)
        q_values = fnn.relu(q_values)
        q_values = fnn.Dense(self.action_dim)(q_values)
        assert_shape(q_values, (batch_size, self.action_dim))

        return q_values


class DoubleCritic(fnn.Module):
    hidden_dim: int
    action_dim: int

    @fnn.compact
    def __call__(self, observations: Array) -> Array:
        """calculate double q

        Args:
            observations (Array): agent observation

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
        qs = VmapCritic(self.hidden_dim, self.action_dim)(observations)
        q1 = qs[0]
        q2 = qs[1]
        return q1, q2


class Actor(fnn.Module):
    hidden_dim: int
    action_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
    ) -> Array:
        """
        calculate agent action distribution

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)

        Returns:
            Array: action distribution
        """
        batch_size = observations.shape[0]

        h_obs = fnn.Dense(self.hidden_dim)(observations)
        h_obs = fnn.relu(h_obs)

        action_logits = fnn.Dense(self.hidden_dim)(h_obs)
        action_logits = fnn.relu(action_logits)
        action_logits = fnn.Dense(self.action_dim)(action_logits)
        action_probs = fnn.softmax(action_logits, axis=-1)
        assert_shape(action_probs, (batch_size, self.action_dim))

        base_dist = tfd.Categorical(probs=action_probs)
        return base_dist
