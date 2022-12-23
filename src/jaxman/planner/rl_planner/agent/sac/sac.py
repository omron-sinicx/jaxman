"""Definition of create and update sac agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Dict, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax.training import checkpoints
from flax.training.train_state import TrainState
from gym.spaces import Box, Dict, Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig

from .actor import create_actor
from .actor import update as update_actor
from .critic import create_critic
from .critic import update as update_critic
from .critic import update_target_critic
from .temperature import create_temp
from .temperature import update as update_temperature


class SAC(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    temperature: TrainState


def create_sac_agent(
    observation_space: Dict,
    action_space: Union[Box, Discrete],
    model_config: DictConfig,
    key: PRNGKey,
) -> Tuple[SAC, PRNGKey]:
    """create sac agent

    Args:
        observation_space (Dict): agent observation space
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
    sac = SAC(actor, critic, target_critic, temp)
    return sac, key


def restore_sac_actor(
    actor: TrainState,
    is_discrete: bool,
    is_diff_drive: bool,
    restore_dir: str,
) -> Tuple[TrainState, TrainState, TrainState, TrainState]:
    """restore pretrained model

    Args:
        actor (TrainState): TrainState of actor
        is_discrete (bool): whether agent has discrete action space
        is_diff_drive (bool): whether agent has diff drive action space
        restore_dir (str): path to restore agent files in.

    Returns:
        Tuple[TrainState]: restored actor
    """
    if not is_discrete:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=actor,
            prefix="continuous_actor",
        ).params
    elif is_diff_drive:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=actor,
            prefix="diff_drive_actor",
        ).params
    else:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=actor,
            prefix="grid_actor",
        ).params
    return actor.replace(params=actor_params)


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
    sac: SAC,
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
        sac.actor,
        sac.critic,
        sac.target_critic,
        sac.temperature,
        batch,
        gamma,
        is_discrete,
    )
    if update_target:
        new_target_critic = update_target_critic(new_critic, sac.target_critic, tau)
    else:
        new_target_critic = sac.target_critic

    if train_actor:
        key, subkey = jax.random.split(key)
        new_actor, actor_info = update_actor(
            subkey, sac.actor, new_critic, sac.temperature, batch, is_discrete
        )
        if auto_temp_tuning:
            new_temp, alpha_info = update_temperature(
                sac.temperature, actor_info["entropy"], target_entropy
            )
        else:
            new_temp = sac.temperature
            alpha = jnp.exp(sac.temperature.params["log_temp"]).astype(float)
            alpha_info = {"temperature": alpha}
        actor_info.update(entropy=actor_info["entropy"].mean())
    else:
        new_actor = sac.actor
        actor_info = {}
        new_temp = sac.temperature
        alpha_info = {}
    new_sac = SAC(new_actor, new_critic, new_target_critic, new_temp)

    return (
        key,
        new_sac,
        {**critic_info, **actor_info, **alpha_info},
    )
