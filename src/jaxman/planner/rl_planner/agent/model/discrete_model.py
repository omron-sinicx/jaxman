"""Definition of models used for discrete SAC

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import flax.linen as fnn
import jax.numpy as jnp
from chex import Array
from flax import linen as fnn
from jaxman.planner.rl_planner.core import AgentObservation

from .base_model import ObsEncoder


class Critic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int
    use_dueling_net: bool

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

        if self.use_dueling_net:
            h1 = fnn.Dense(self.hidden_dim)(h)
            value = fnn.Dense(1)(h1)
            h2 = fnn.Dense(self.hidden_dim)(h)
            adv = fnn.Dense(self.action_dim)(h2)
            adv_scaled = adv - jnp.mean(adv, axis=-1, keepdims=True)
            q_values = value - adv_scaled
        else:
            h = fnn.Dense(self.hidden_dim)(h)
            h = fnn.relu(h)
            q_values = fnn.Dense(self.action_dim)(h)

        return q_values


class DoubleCritic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int
    use_dueling_net: bool

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
        qs = VmapCritic(
            self.hidden_dim, self.msg_dim, self.action_dim, self.use_dueling_net
        )(
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
            Array: action probability
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

        return action_probs


class MultiHeadCritic(fnn.Module):
    hidden_dim: int
    msg_dim: int
    action_dim: int
    use_dueling_net: bool

    @fnn.compact
    def __call__(self, observations: AgentObservation) -> Array:
        """
        multi head q value
        branch 1 compute q value for item-carrying agent
        branch 2 compute q value for not item-carrying agent

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication

        Returns:
            Array: q-value according to the agent's state (whether it is carrying items or not)
        """
        VmapCritic = fnn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        qs = VmapCritic(
            self.hidden_dim, self.msg_dim, self.action_dim, self.use_dueling_net
        )(
            observations,
        )
        q1 = qs[0]
        q2 = qs[1]

        q = (
            observations.is_hold_item * q1
            + jnp.logical_not(observations.is_hold_item) * q2
        )

        return q
