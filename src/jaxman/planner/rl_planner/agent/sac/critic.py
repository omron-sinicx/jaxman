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
from gym.spaces import Box, Dict, Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig

from ..model.continuous_model import DoubleCritic as ContinuousCritic
from ..model.discrete_model import DoubleCritic as DiscreteCritic


def create_critic(
    observation_space: Dict,
    action_space: Union[Box, Discrete],
    config: DictConfig,
    key: PRNGKey,
) -> TrainState:
    """
    create actor TrainState

    Args:
        observation_space (Dict): observation space
        action_space (Box): action space
        config (DictConfig): configuration of actor
        key (PRNGKey): PRNGKey for actor

    Returns:
        TrainState: actor TrainState
    """
    obs_dim = observation_space["obs"].shape
    comm_dim = observation_space["comm"].shape
    mask_dim = observation_space["mask"].shape
    if isinstance(action_space, Box):
        action_dim = action_space.shape[0]
        critic_fn = ContinuousCritic(config.hidden_dim, config.msg_dim)
        params = critic_fn.init(
            key,
            jnp.ones([1, *obs_dim]),
            jnp.ones([1, *comm_dim]),
            jnp.ones([1, *mask_dim]),
            jnp.ones([1, action_dim]),
        )["params"]
    else:
        action_dim = action_space.n
        critic_fn = DiscreteCritic(config.hidden_dim, config.msg_dim, action_dim)
        params = critic_fn.init(
            key,
            jnp.ones([1, *obs_dim]),
            jnp.ones([1, *comm_dim]),
            jnp.ones([1, *mask_dim]),
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
        batch_size = batch.observations.shape[0]

        next_dist = actor.apply_fn(
            {"params": actor.params},
            batch.next_observations,
            batch.next_communications,
            batch.next_neighbor_masks,
        )

        next_action_probs = next_dist.probs
        z = next_action_probs == 0.0
        next_action_probs += z.astype(float) * 1e-8
        next_log_probs = jnp.log(next_action_probs)

        next_q1, next_q2 = target_critic.apply_fn(
            {"params": target_critic.params},
            batch.next_observations,
            batch.next_communications,
            batch.next_neighbor_masks,
        )
        next_q = jnp.minimum(next_q1, next_q2)
        temp = temperature.apply_fn({"params": temperature.params})

        next_v = jnp.sum(
            (next_action_probs * (next_q - temp * next_log_probs)), axis=-1
        )
        assert_shape(next_v, (batch_size,))
        target_q = batch.rewards + gamma * batch.masks * next_v
        assert_shape(target_q, (batch_size,))

        q1, q2 = critic.apply_fn(
            {"params": params},
            batch.observations,
            batch.communications,
            batch.neighbor_masks,
        )
        q1 = jax.vmap(lambda q_values, i: q_values[i])(q1, batch.actions)
        q2 = jax.vmap(lambda q_values, i: q_values[i])(q2, batch.actions)
        td_error = (q1.reshape(batch_size) - target_q) ** 2 + (
            q2.reshape(batch_size) - target_q
        ) ** 2

        critic_loss = td_error.mean()
        assert_shape(td_error, (batch_size,))
        return critic_loss

    def continuous_loss_fn(params: FrozenDict) -> Array:
        batch_size = batch.observations.shape[0]

        dist = actor.apply_fn(
            {"params": actor.params},
            batch.next_observations,
            batch.next_communications,
            batch.next_neighbor_masks,
        )
        next_actions = dist.sample(seed=key)
        next_log_probs = dist.log_prob(next_actions)

        next_q1, next_q2 = target_critic.apply_fn(
            {"params": target_critic.params},
            batch.next_observations,
            batch.next_communications,
            batch.next_neighbor_masks,
            next_actions,
        )
        next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))

        temp = temperature.apply_fn({"params": temperature.params})
        target_q = batch.rewards + gamma * batch.masks * (
            next_q - next_log_probs * temp
        )

        q1, q2 = critic.apply_fn(
            {"params": params},
            batch.observations,
            batch.communications,
            batch.neighbor_masks,
            batch.actions,
        )
        td_error = (jnp.squeeze(q1) - target_q) ** 2 + (jnp.squeeze(q2) - target_q) ** 2
        assert_shape(td_error, (batch_size,))
        critic_loss = td_error.mean()
        return critic_loss

    if is_discrete:
        grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=False)
    else:
        grad_fn = jax.value_and_grad(continuous_loss_fn, has_aux=False)
    loss, grads = grad_fn(critic.params)
    critic = critic.apply_gradients(grads=grads)
    return critic, {"critic_loss": loss}
