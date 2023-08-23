"""jax sac critic creater and update

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
from ..model.continuous_model import DoubleCritic as ContinuousCritic
from ..model.discrete_model import DoubleCritic as DiscreteCritic

tfd = tfp.distributions


def create_critic(
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
        critic_fn = ContinuousCritic(config.hidden_dim, config.msg_dim)
        params = critic_fn.init(
            key,
            dummy_observations,
            jnp.ones([1, action_dim]),
        )["params"]
    else:
        action_dim = action_space.n
        critic_fn = DiscreteCritic(config.hidden_dim, config.msg_dim, action_dim)
        params = critic_fn.init(
            key,
            dummy_observations,
        )["params"]

    lr_rate_schedule = optax.cosine_decay_schedule(
        config.actor_lr, config.decay_steps, config.decay_alpha
    )
    tx = optax.adam(learning_rate=lr_rate_schedule)
    critic = TrainState.create(apply_fn=critic_fn.apply, params=params, tx=tx)
    return critic


@jax.jit
def update_target_critic(
    critic: TrainState, target_critic: TrainState, tau: float
) -> TrainState:
    """
    update target critic

    Args:
        critic (TrainState): TrainState of critic
        target_critic (TrainState): TrainState of target_critic
        tau (float): update ratio

    Returns:
        TrainState: TrainState of updated target_critic
    """
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=("is_discrete"))
def update(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temperature: TrainState,
    batch: TrainBatch,
    gamma: float,
    is_discrete: bool,
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

        next_action_probs = actor.apply_fn(
            {"params": actor.params},
            batch.next_observations,
        )

        z = next_action_probs == 0.0
        next_action_probs += z.astype(float) * 1e-8
        next_log_probs = jnp.log(next_action_probs)

        next_q1, next_q2 = target_critic.apply_fn(
            {"params": target_critic.params},
            batch.next_observations,
        )
        next_q = jnp.minimum(next_q1, next_q2)
        temp = jnp.exp(temperature.apply_fn({"params": temperature.params}))

        next_v = jnp.sum(
            (next_action_probs * (next_q - temp * next_log_probs)), axis=-1
        )
        assert_shape(next_v, (batch_size,))
        target_q = batch.rewards + gamma * batch.masks * next_v
        assert_shape(target_q, (batch_size,))

        q1, q2 = critic.apply_fn(
            {"params": params},
            batch.observations,
        )
        q1 = jax.vmap(lambda q_values, i: q_values[i])(q1, batch.actions)
        q2 = jax.vmap(lambda q_values, i: q_values[i])(q2, batch.actions)
        critic_loss = (q1.reshape(batch_size) - target_q) ** 2 + (
            q2.reshape(batch_size) - target_q
        ) ** 2

        assert_shape(critic_loss, (batch_size,))
        return critic_loss.mean()

    def continuous_loss_fn(params: FrozenDict) -> Array:
        batch_size = batch.observations.base_observation.shape[0]

        means, log_stds = actor.apply_fn(
            {"params": actor.params},
            batch.next_observations,
        )
        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)

        next_q1, next_q2 = target_critic.apply_fn(
            {"params": target_critic.params},
            batch.next_observations,
            next_actions,
        )
        next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))

        temp = jnp.exp(temperature.apply_fn({"params": temperature.params}))
        target_q = batch.rewards + gamma * batch.masks * (
            next_q - next_log_probs * temp
        )

        q1, q2 = critic.apply_fn(
            {"params": params},
            batch.observations,
            batch.actions,
        )
        critic_loss = (jnp.squeeze(q1) - target_q) ** 2 + (
            jnp.squeeze(q2) - target_q
        ) ** 2

        assert_shape(critic_loss, (batch_size,))
        return critic_loss.mean()

    if is_discrete:
        grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=False)
    else:
        grad_fn = jax.value_and_grad(continuous_loss_fn, has_aux=False)
    loss, grads = grad_fn(critic.params)
    critic = critic.apply_gradients(grads=grads)
    return critic, {
        "critic_loss": loss,
    }
