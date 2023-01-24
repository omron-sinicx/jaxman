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
from gym.spaces import Box
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig
from tensorflow_probability.substrates import jax as tfp

from ...core import AgentObservation
from ..model.continuous_model import Actor as ContinuousActor
from ..model.discrete_model import Actor as DiscreteActor

tfd = tfp.distributions


def create_actor(
    observation_space: GymDict,
    action_space: Union[Box, Discrete],
    config: DictConfig,
    key: PRNGKey,
) -> TrainState:
    """
    create actor TrainState

    Args:
        observation_space (GymDict): observation space
        action_space (Box): action space
        config (DictConfig): configuration of actor
        key (PRNGKey): PRNGKey for actor

    Returns:
        TrainState: actor TrainState
    """
    obs = jnp.ones([1, *observation_space["obs"].shape])
    comm = jnp.ones([1, *observation_space["comm"].shape])
    mask = jnp.ones([1, *observation_space["mask"].shape])
    if observation_space["item_pos"].shape[0] > 0:
        item_pos = jnp.ones([1, *observation_space["item_pos"].shape])
        item_mask = jnp.ones([1, *observation_space["item_mask"].shape])
    else:
        item_pos = item_mask = None
    dummy_observations = AgentObservation(obs, comm, mask, item_pos, item_mask)

    if isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        actor_fn = ContinuousActor(config.hidden_dim, config.msg_dim, action_dim)
    else:
        action_dim = action_space.n
        actor_fn = DiscreteActor(config.hidden_dim, config.msg_dim, action_dim)

    params = actor_fn.init(key, dummy_observations)["params"]

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

        batch_size = batch.observations.base_observation.shape[0]
        action_probs = actor.apply_fn(
            {"params": actor_params},
            batch.observations,
        )

        z = action_probs == 0.0
        action_probs += z.astype(float) * 1e-8
        log_probs = jnp.log(action_probs)
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        q1, q2 = critic.apply_fn(
            {"params": critic.params},
            batch.observations,
        )
        q = jnp.minimum(q1, q2)
        temp = jnp.exp(temperature.apply_fn({"params": temperature.params}))
        actor_loss = -(jnp.sum(action_probs * q, axis=-1) + temp * entropy)
        assert_shape(actor_loss, (batch_size,))
        assert_shape(entropy, (batch_size,))
        actor_loss = actor_loss.mean()
        return actor_loss, entropy

    def continuous_loss_fn(actor_params: FrozenDict) -> Tuple[Array, Dict]:
        batch_size = batch.observations.shape[0]
        means, log_stds = actor.apply_fn(
            {"params": actor_params},
            batch.observations,
        )
        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        q1, q2 = critic.apply_fn(
            {"params": critic.params},
            batch.observations,
            actions,
        )

        q = jnp.squeeze(jnp.minimum(q1, q2))

        temp = jnp.exp(temperature.apply_fn({"params": temperature.params}))
        actor_loss = log_probs * temp - q
        assert_shape(actor_loss, (batch_size,))
        assert_shape(log_probs, (batch_size,))
        actor_loss = actor_loss.mean()
        return actor_loss, -log_probs

    if is_discrete:
        grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=True)
    else:
        grad_fn = jax.value_and_grad(continuous_loss_fn, has_aux=True)
    (actor_loss, entropy), grads = grad_fn(actor.params)
    actor = actor.apply_gradients(grads=grads)
    info = {"actor_loss": actor_loss, "entropy": entropy}
    return actor, info
