from functools import partial
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey, assert_shape
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from gym.spaces import Box
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig
from tensorflow_probability.substrates import jax as tfp

from ...core import AgentObservation
from ..model.continuous_model import DoubleCritic as ContinuousCritic
from ..model.discrete_model import DoubleCritic as DiscreteCritic

tfd = tfp.distributions
tfb = tfp.bijectors

@partial(jax.jit, static_argnames=("is_discrete"))
def update(
    actor: TrainState,
    target_actor: TrainState,
    batch: TrainBatch,
    gamma: float,
) -> Tuple[TrainState, Dict]:
    """update critic

    Args:
        key (PRNGKey): random variables key
        actor (TrainState): TrainState of actor
        critic (TrainState): TrainState of critic
        target_critic (TrainState): TrainState of target critic
        temperature (TrainState): TrainState of temperature
        batch (TrainBatch): train batch
        gamma (float): decay rate
        is_discrete (bool): whether agent action space is Discrete or not

    Returns:
        Tuple[TrainState, InfoDict, Array]: TrainState of updated critic, loss indicators
    """

    def discrete_loss_fn(params: FrozenDict) -> Array:
        batch_size = batch.observations.base_observation.shape[0]

        next_q = target_actor.apply_fn(
            {"params": target_actor.params},
            batch.next_observations,
        )

        target_q = batch.rewards + batch.masks * gamma * jnp.max(next_q,-1)
        assert_shape(target_q, (batch_size,))

        q = actor.apply_fn(
            {"params": params}, batch.observations
        )
        q = jax.vmap(lambda q_values, i: q_values[i])(q, batch.actions)

        critic_loss = (q.reshape(batch_size) - target_q) ** 2
        assert_shape(critic_loss, (batch_size,))
        return critic_loss.mean()

    grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=False)
    loss, grads = grad_fn(actor.params)
    actor = actor.apply_gradients(grads=grads)
    return actor, {"critic_loss": loss}