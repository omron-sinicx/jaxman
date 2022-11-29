"""Definition of create and update sac agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax.training.train_state import TrainState
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig

from .actor import create_actor
from .actor import update as update_actor
from .critic import create_critic
from .critic import update as update_critic
from .critic import update_target_critic
from .temperature import create_temp
from .temperature import update as update_temperature


def create_sac_agent(
    observation_space: Box,
    action_space: Union[Box, Discrete],
    model_config: DictConfig,
    key: PRNGKey,
) -> Tuple[TrainState, TrainState, TrainState, TrainState, PRNGKey]:
    """create sac agent

    Args:
        observation_space (Box): agent observation space
        action_space (Union[Box, Discrete]): agent action space
        model_config (DictConfig): model configurations
        key (PRNGKey): random variable key

    Returns:
        Tuple[TrainState,TrainState,TrainState,TrainState,PRNGKey]: SAC agent and key
    """
    key, actor_key, critic_key, temp_key = jax.random.split(key, 4)
    actor = create_actor(observation_space, action_space, model_config, actor_key)
    critic = create_critic(observation_space, action_space, model_config, critic_key)
    target_critic = create_critic(
        observation_space, action_space, model_config, critic_key
    )
    temp = create_temp(model_config, temp_key)
    return actor, critic, target_critic, temp, key


@partial(
    jax.jit,
    static_argnames=(
        "is_discrete",
        "auto_temp_tuning",
        "update_target",
        "train_actor",
    ),
)
def _update_sac_jit(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: TrainBatch,
    gamma: float,
    tau: float,
    is_discrete: bool,
    target_entropy: float,
    auto_temp_tuning: bool,
    update_target: bool,
    train_actor: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]:
    """
    update SAC agent network

    Args:
        key (PRNGKey): random varDable key
        actor (TrainState): TrainState of actor
        critic (TrainState): TrainState of critic
        target_critic (TrainState): TrainState of target_critic
        temp (TrainState): TrainState of temperature
        batch (TrainBatch): Train Batch
        gamma (float): gamma. decay rate
        tau (float): tau. target critic update rate
        is_discrete (bool): whether agent action space is Discrete or not
        target_entropy (float): target entropy
        auto_temp_tuning (bool): whether to update temperature
        update_target (bool): whether to update target_critic network
        train_actor (bool): whether to update actor

    Returns:
        Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]: new key, updated SAC agent, loss informations
    """

    key, subkey = jax.random.split(key)
    new_critic, critic_info = update_critic(
        subkey,
        actor,
        critic,
        target_critic,
        temp,
        batch,
        gamma,
        is_discrete,
    )
    if update_target:
        new_target_critic = update_target_critic(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    if train_actor:
        key, subkey = jax.random.split(key)
        new_actor, actor_info = update_actor(
            subkey, actor, new_critic, temp, batch, is_discrete
        )
        if auto_temp_tuning:
            new_temp, alpha_info = update_temperature(
                temp, actor_info["entropy"], target_entropy
            )
        else:
            new_temp = temp
            alpha = jnp.exp(temp.params["log_temp"]).astype(float)
            alpha_info = {"temperature": alpha}
    else:
        new_actor = actor
        actor_info = {}
        new_temp = temp
        alpha_info = {}

    return (
        key,
        new_actor,
        new_critic,
        new_target_critic,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )
