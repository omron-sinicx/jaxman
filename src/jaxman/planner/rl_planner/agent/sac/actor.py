"""jax sac actor creater and update

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey, assert_shape
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig

from ..model.continuous_model import Actor as ContinuousActor
from ..model.discrete_model import Actor as DiscreteActor


def create_actor(
    observation_space: Box,
    action_space: Union[Box, Discrete],
    config: DictConfig,
    key: PRNGKey,
) -> TrainState:
    """
    create actor TrainState

    Args:
        observation_space (Box): observation_space of single agent
        action_space (Box): action_space of single agent
        config (DictConfig): configuration of actor
        key (PRNGKey): PRNGKey for actor

    Returns:
        TrainState: actor TrainState
    """
    obs_dim = observation_space.shape[0]
    if isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        actor_fn = ContinuousActor(config.hidden_dim, action_dim)
    else:
        action_dim = action_space.n
        actor_fn = DiscreteActor(config.hidden_dim, action_dim)

    params = actor_fn.init(key, jnp.ones([1, obs_dim]))["params"]

    lr_rate_schedule = optax.cosine_decay_schedule(
        config.actor_lr, config.decay_steps, config.decay_alpha
    )
    tx = optax.adam(learning_rate=lr_rate_schedule)
    actor = TrainState.create(apply_fn=actor_fn.apply, params=params, tx=tx)
    return actor


@partial(jax.jit, static_argnames=("is_discrete"))
def update(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    temperature: TrainState,
    batch: TrainBatch,
    is_discrete: bool,
) -> Tuple[TrainState, Dict]:
    """
    update actor network

    Args:
        key (PRNGKey): random variables key
        actor (TrainState): TrainState of actor
        critic (TrainState): TrainState of critic
        temperature (TrainState): TrainState of temperature
        batch (TrainBatch): batched agent's experience
        is_discrete (bool): whether agent action space is Discrete or not

    Returns:
        Tuple[TrainState, Dict]: updated actor, loss information
    """

    def discrete_loss_fn(actor_params: FrozenDict) -> Tuple[Array, Dict]:

        batch_size = batch.observations.shape[0]
        dist = actor.apply_fn(
            {"params": actor_params},
            batch.observations,
        )

        action_probs = dist.probs
        z = action_probs == 0.0
        action_probs += z.astype(float) * 1e-8
        log_probs = jnp.log(action_probs)
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        q1, q2 = critic.apply_fn({"params": critic.params}, batch.observations)
        q = jnp.minimum(q1, q2)
        temp = temperature.apply_fn({"params": temperature.params})
        actor_loss = jnp.sum((action_probs * (temp * log_probs - q)), axis=-1)
        assert_shape(actor_loss, (batch_size,))
        actor_loss = actor_loss.mean()
        return actor_loss, entropy.mean()

    def continuous_loss_fn(actor_params: FrozenDict) -> Tuple[Array, Dict]:
        batch_size = batch.observations.shape[0]
        dist = actor.apply_fn(
            {"params": actor_params},
            batch.observations,
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        q1, q2 = critic.apply_fn({"params": critic.params}, batch.observations, actions)
        q = jnp.squeeze(jnp.minimum(q1, q2))

        temp = temperature.apply_fn({"params": temperature.params})
        actor_loss = log_probs * temp - q
        assert_shape(actor_loss, (batch_size,))
        actor_loss = actor_loss.mean()
        return actor_loss, -log_probs.mean()

    if is_discrete:
        grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=True)
    else:
        grad_fn = jax.value_and_grad(continuous_loss_fn, has_aux=True)
    (actor_loss, entropy), grads = grad_fn(actor.params)
    actor = actor.apply_gradients(grads=grads)
    info = {"actor_loss": actor_loss, "entropy": entropy}
    return actor, info