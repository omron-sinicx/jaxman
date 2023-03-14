"""upate function for Deep Q Network agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, assert_shape
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@partial(
    jax.jit,
    static_argnames=(
        "is_pal",
        "use_maxmin_dqn",
        "N",
        "use_ddqn",
        "use_k_step_learning",
    ),
)
def update(
    key: PRNGKey,
    actor: TrainState,
    target_actor: TrainState,
    batch: TrainBatch,
    gamma: float,
    is_pal: bool,
    alpha: float,
    use_maxmin_dqn: bool,
    N: int,
    use_ddqn: bool,
    use_k_step_learning: bool,
    k: int,
) -> Tuple[TrainState, Array, Dict]:
    """update dqn_actor

    Args:
        key (PRNGkey): random variable key
        actor (TrainState): TrainState of actor
        target_actor (TrainState): TrainState of target_actor
        batch (TrainBatch): train batch
        gamma (float): decay rate
        is_pal (bool): whether to use persistent advantage laerning or not
        alpha (float): weight of action gap used for persistent advantage learning
        use_maxmin_dqn (bool): whether to use Maxmin Q-Learning
        N (int): number of networks for Maxmin Q-Learning
        use_ddqn (bool): whether to use double dqn
        use_k_step_learning (bool): whether to use k step learning
        k (int): k for multi step learning

    Returns:
        Tuple[TrainState, Array, Dict]: TrainState of updated dqn_actor, priority, loss indicators
    """

    def discrete_loss_fn(params: FrozenDict) -> Array:
        batch_size = batch.observations.base_observation.shape[0]

        next_q = target_actor.apply_fn(
            {"params": target_actor.params},
            batch.next_observations,
        )
        if use_maxmin_dqn:
            next_q = jnp.min(next_q, axis=1)
        if use_ddqn:
            next_action = jnp.argmax(
                actor.apply_fn(
                    {"params": params},
                    batch.next_observations,
                ),
                -1,
            )
            next_q = jax.vmap(lambda q_values, i: q_values[i])(next_q, next_action)
        else:
            next_q = jnp.max(next_q, -1)

        if use_k_step_learning:
            target_q = batch.rewards + batch.masks * (gamma**k) * next_q
        else:
            target_q = batch.rewards + batch.masks * gamma * next_q

        assert_shape(target_q, (batch_size,))

        if is_pal:
            q_s = target_actor.apply_fn(
                {"params": target_actor.params},
                batch.observations,
            )
            q_s_a = jax.vmap(lambda q_values, i: q_values[i])(q_s, batch.actions)
            action_gap = jnp.max(q_s, -1) - q_s_a
            # next state
            next_q = actor.apply_fn(
                {"params": params},
                batch.next_observations,
            )
            next_v = jnp.max(next_q, -1)
            next_action = jnp.argmax(
                target_actor.apply_fn(
                    {"params": target_actor.params},
                    batch.next_observations,
                ),
                axis=-1,
            )
            next_q_s_a = jax.vmap(lambda q_values, i: q_values[i])(next_q, next_action)
            next_action_gap = jax.lax.stop_gradient(next_v - next_q_s_a)

            target_q = target_q - alpha * jnp.minimum(action_gap, next_action_gap)

        q_values = actor.apply_fn({"params": params}, batch.observations)
        if use_maxmin_dqn:
            subnet_id = jax.random.choice(key, N, shape=(1,))
            q_value = jax.vmap(
                lambda q_values, i, k: q_values[k, i], in_axes=[0, 0, None]
            )(q_values, batch.actions, subnet_id)
        else:
            q_value = jax.vmap(lambda q_values, i: q_values[i], in_axes=[0, 0])(
                q_values, batch.actions
            )
        td_error = q_value.reshape(batch_size) - target_q
        critic_loss = td_error**2
        weight_critic_loss = critic_loss

        # for evaluation
        prob = jax.nn.softmax(q_values) + 1e-10
        entropy = -jnp.mean(jnp.sum(prob * jnp.log(prob), -1))

        return weight_critic_loss.mean(), (jnp.abs(td_error), entropy)

    grad_fn = jax.value_and_grad(discrete_loss_fn, has_aux=True)
    (loss, (priority, entropy)), grads = grad_fn(actor.params)
    actor = actor.apply_gradients(grads=grads)
    return actor, priority, {"critic_loss": loss, "entropy": entropy}
