"""Definition of basic model structure for SAC Agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
import flax.linen as fnn
import jax
import jax.numpy as jnp
from chex import Array, assert_shape
from flax import linen as fnn


@jax.jit
def msg_attention(query: Array, key: Array, value: Array) -> Array:
    """
    compute attention

    Args:
        query (Array): query. shape: (batch_size, msg_dim)
        key (Array): key. shape: (batch_size, num_comm_agents, msg_dim)
        value (Array): value. shape: (batch_size, num_comm_agents, msg_dim)

    Returns:
        Array: attentioned message
    """
    weight = jax.vmap(jnp.matmul)(key, query) / key.shape[-1]
    weight = fnn.softmax(weight)
    weighted_value = jax.vmap(jnp.dot)(weight, value)
    return weighted_value


class ObsActEncoder(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
        communications: Array,
        actions: Array,
    ) -> Array:
        """
        encode observation, communication, action

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)
            communications (Array): communications with neighbor agents. shape: (batch_size, num_comm_agents, comm_dim)
            actions (Array): agent actions. shape: (batch_size, action_dim)

        Returns:
            Array: computed hidden state
        """
        batch_size = observations.shape[0]
        num_comm_agents = communications.shape[1]

        inputs = jnp.concatenate([observations, actions], axis=-1)

        h_obs = fnn.Dense(self.hidden_dim)(inputs)
        h_obs = fnn.relu(h_obs)
        h_comm = fnn.Dense(self.hidden_dim)(communications)
        h_comm = fnn.relu(h_comm)

        # communication
        query = fnn.Dense(self.msg_dim)(h_obs)
        key = fnn.Dense(self.msg_dim)(h_comm)
        value = fnn.Dense(self.msg_dim)(h_comm)
        assert_shape(query, (batch_size, self.msg_dim))
        assert_shape(key, (batch_size, num_comm_agents, self.msg_dim))

        # attention
        h_comm = msg_attention(query, key, value)
        assert_shape(h_comm, (batch_size, self.msg_dim))

        h_obs_comm = jnp.concatenate((h_obs, h_comm), -1)
        h_obs_comm = fnn.Dense(self.hidden_dim)(h_obs_comm)
        h_obs_comm = fnn.relu(h_obs_comm)

        return h_obs_comm


class ObsEncoder(fnn.Module):
    hidden_dim: int
    msg_dim: int

    @fnn.compact
    def __call__(
        self,
        observations: Array,
        communications: Array,
    ) -> Array:
        """
        encode observation, communication

        Args:
            observations (Array): observations. shape: (batch_size, obs_dim)
            communications (Array): communications with neighbor agents. shape: (batch_size, num_comm_agents, comm_dim)

        Returns:
            Array: compute hidden state
        """
        batch_size = observations.shape[0]
        num_comm_agents = communications.shape[1]

        h_obs = fnn.Dense(self.hidden_dim)(observations)
        h_obs = fnn.relu(h_obs)
        h_comm = fnn.Dense(self.hidden_dim)(communications)
        h_comm = fnn.relu(h_comm)

        # communication
        query = fnn.Dense(self.msg_dim)(h_obs)
        key = fnn.Dense(self.msg_dim)(h_comm)
        value = fnn.Dense(self.msg_dim)(h_comm)
        assert_shape(query, (batch_size, self.msg_dim))
        assert_shape(key, (batch_size, num_comm_agents, self.msg_dim))

        # attention
        h_comm = msg_attention(query, key, value)
        assert_shape(h_comm, (batch_size, self.msg_dim))

        h_obs_comm = jnp.concatenate((h_obs, h_comm), -1)
        h_obs_comm = fnn.Dense(self.hidden_dim)(h_obs_comm)
        h_obs_comm = fnn.relu(h_obs_comm)

        return h_obs_comm
