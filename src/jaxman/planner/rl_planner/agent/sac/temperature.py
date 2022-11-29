""" jax sac temperature creater and update

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


class Temperature(nn.Module):
    initial_temperature: float = 0.05

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """
        return temperature

        Returns:
            jnp.ndarray: temperature
        """
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


def create_temp(config, key):
    """
    create temperature TrainState

    Args:
        config (DictConfig): configuration of temperature
        key (PRNGKey): PRNGKey for temperature

    Returns:
        TrainState: temperature TrainState
    """
    temp_fn = Temperature()
    params = temp_fn.init(key)["params"]
    lr_rate_schedule = optax.cosine_decay_schedule(
        config.critic_lr, config.decay_steps, config.decay_alpha
    )
    tx = optax.adam(learning_rate=lr_rate_schedule)
    temp = TrainState.create(apply_fn=temp_fn.apply, params=params, tx=tx)
    return temp


@jax.jit
def update(
    temp: TrainState, entropy: float, target_entropy: float
) -> Tuple[TrainState, Dict]:
    """
    update temperature

    Args:
        temp (TrainState): TrainState of temperature
        entropy (float): entropy
        target_entropy (float): target entropy

    Returns:
        Tuple[TrainState, Dict]: TrainState of updated temperature, loss indicator
    """

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({"params": temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss

    grad_fn = jax.value_and_grad(temperature_loss_fn, has_aux=False)
    loss, grads = grad_fn(temp.params)
    temp = temp.apply_gradients(grads=grads)
    alpha = jnp.exp(temp.params["log_temp"]).astype(float)
    return temp, {"temperature_loss": loss, "temperature": alpha}
