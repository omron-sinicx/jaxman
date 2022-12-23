"""Definition of sample agent action

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax import linen as fnn
from flax.core.frozen_dict import FrozenDict
from jaxman.env import AgentObservation
from jaxman.planner.rl_planner.core import AgentObservation as ModelInput
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def build_sample_agent_action(actor_fn: Callable, is_discrete: bool, evaluate: bool):
    def sample_action(
        params: FrozenDict,
        observations: AgentObservation,
        key: PRNGKey,
    ) -> Tuple[PRNGKey, Array]:
        """sample agent action

        Args:
            params (FrozenDict): agent parameter
            observations (Array): agent observatoin
            key (PRNGKey): random key variable

        Returns:
            Tuple[PRNGKey, Array]: new key, sampled action
        """
        obs, comm, mask, item_pos, item_mask = observations.split_observation()
        obs = ModelInput(obs, comm, mask, item_pos, item_mask)
        # if evaluate:
        #     if is_discrete:
        #         action_probs = actor_fn({"params": params}, obs)
        #         actions = jnp.argmax(action_probs, axis=-1)
        #     else:
        #         means, log_stds = actor_fn({"params": params}, obs)
        #         actions = means
        # else:
        #     if is_discrete:
        #         action_probs = actor_fn({"params": params}, obs)
        #         action_dist = tfd.Categorical(probs=action_probs)
        #     else:
        #         means, log_stds = actor_fn({"params": params}, obs)
        #         action_dist = tfd.MultivariateNormalDiag(
        #             loc=means, scale_diag=jnp.exp(log_stds)
        #         )

        #     subkey, key = jax.random.split(key)
        #     actions = action_dist.sample(seed=subkey)
        q_values = actor_fn({"params": params}, obs)
        if evaluate:
            actions = jnp.argmax(q_values, axis=-1)
        else:
            key, key1, key2 = jax.random.split(key, 3)
            action_prob = fnn.softmax(q_values, axis=-1)
            actions = tfd.Categorical(action_prob).sample(seed=key1)
            # epsilon = jax.random.uniform(key2, shape=(1,))
            # rand_action = jax.random.choice(key2, actions.shape[0], shape=(1,))
            # random = epsilon < 0.05
            # actions = actions * jnp.logical_not(random) + rand_action * random

        return key, actions

    return jax.jit(sample_action)
