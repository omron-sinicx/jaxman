"""Definition of sample agent action

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentObservation


def build_sample_agent_action(actor_fn: Callable, is_discrete: bool, evaluate: bool):
    def sample_action(
        params: FrozenDict, observations: AgentObservation, key: PRNGKey
    ) -> Tuple[PRNGKey, Array]:
        """sample agent action

        Args:
            params (FrozenDict): agent parameter
            observations (Array): agent observatoin
            key (PRNGKey): random key variable

        Returns:
            Tuple[PRNGKey, Array]: new key, sampled action
        """
        obs, comm = observations.split_obs_comm()
        action_dist = actor_fn({"params": params}, obs, comm)
        if evaluate:
            if is_discrete:
                actions = jnp.argmax(action_dist.probs, axis=-1)
            else:
                actions = action_dist.mean()
        else:
            subkey, key = jax.random.split(key)
            actions = action_dist.sample(seed=subkey)
        return key, actions

    return jax.jit(sample_action)
