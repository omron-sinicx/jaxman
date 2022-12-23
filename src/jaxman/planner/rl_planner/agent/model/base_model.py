"""Definition of basic model structure for SAC Agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
import flax.linen as fnn
import jax
import jax.numpy as jnp
from chex import Array
from flax import linen as fnn

from ...core import AgentObservation


@jax.jit
def msg_attention(
    query: Array, key: Array, value: Array, neighbor_mask: Array
) -> Array:
    """
    compute attention

    Args:
        query (Array): query. shape: (batch_size, msg_dim)
        key (Array): key. shape: (batch_size, num_comm_agents, msg_dim)
        value (Array): value. shape: (batch_size, num_comm_agents, msg_dim)
        neighbor_mask (Array): mask for obtaining only neighboring agent communications. shape: (batch_size, num_agents)

    Returns:
        Array: attentioned message
    """
    weight = jax.vmap(jnp.matmul)(key, query) / key.shape[-1]
    masked_weight = weight * neighbor_mask
    masked_weight = fnn.softmax(masked_weight)
    weighted_value = jax.vmap(jnp.dot)(masked_weight, value)
    return weighted_value


class AttentionBlock(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(self, h_obs, comm, mask):
        h_comm = fnn.Dense(self.hidden_dim)(comm)
        h_comm = fnn.relu(h_comm)

        query = fnn.Dense(self.msg_dim)(h_obs)
        key = fnn.Dense(self.msg_dim)(h_comm)
        value = fnn.Dense(self.msg_dim)(h_comm)

        weighted_value = msg_attention(query, key, value, mask)
        return weighted_value


class ObsActEncoder(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: AgentObservation,
        actions: Array,
    ) -> Array:
        """
        encode observation, communication, action

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication
            actions (Array): agent actions. shape: (batch_size, action_dim)

        Returns:
            Array: computed hidden state
        """
        inputs = jnp.concatenate([observations.base_observation, actions], axis=-1)

        h_obs = fnn.Dense(self.hidden_dim)(inputs)
        h_obs = fnn.relu(h_obs)

        # attention agent communication
        comm_attention_block = AttentionBlock(self.hidden_dim, self.msg_dim)
        h_comm = comm_attention_block(
            h_obs, observations.communication, observations.agent_mask
        )
        h = jnp.concatenate((h_obs, h_comm), -1)

        if observations.item_positions is not None:
            item_attention_block = AttentionBlock(self.hidden_dim, self.msg_dim)
            h_item = item_attention_block(
                h_obs,
                observations.item_positions,
                observations.item_mask,
            )

            h = jnp.concatenate((h, h_item), -1)

        h = fnn.Dense(self.hidden_dim)(h)
        h = fnn.relu(h)

        return h


class ObsEncoder(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: AgentObservation,
    ) -> Array:
        """
        encode observation, communication

        Args:
            observations (AgentObservation): NamedTuple for observation of agent. consisting of basic observations and communication

        Returns:
            Array: compute hidden state
        """

        h_obs = fnn.Dense(self.hidden_dim)(observations.base_observation)
        h_obs = fnn.relu(h_obs)

        # attention agent communication
        comm_attention_block = AttentionBlock(self.hidden_dim, self.msg_dim)
        h_comm = comm_attention_block(
            h_obs, observations.communication, observations.agent_mask
        )
        h = jnp.concatenate((h_obs, h_comm), -1)

        if observations.item_positions is not None:
            item_attention_block = AttentionBlock(self.hidden_dim, self.msg_dim)
            h_item = item_attention_block(
                h_obs,
                observations.item_positions,
                observations.item_mask,
            )
            print(h_item.shape)
            h = jnp.concatenate((h, h_item), -1)

        h = fnn.Dense(self.hidden_dim)(h)
        h = fnn.relu(h)

        return h
