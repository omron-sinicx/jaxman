"""Definition of create and update sac agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from gym.spaces import Box, Dict, Discrete
from jaxman.env import AgentObservation
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig
from tensorflow_probability.substrates import jax as tfp

from .actor import create_actor
from .actor import update as update_actor
from .critic import create_critic
from .critic import update as update_critic
from .critic import update_target_critic
from .temperature import create_temp
from .temperature import update as update_temperature

tfd = tfp.distributions


class SAC(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_network: TrainState
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
        Tuple[SAC,PRNGKey]: SAC agent and key
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
) -> Tuple[PRNGKey, SAC, Array, Dict]:
    """
    update SAC agent network

    Args:
        key (PRNGKey): random varDable key
        sac (SAC): Namedtuple stores sac agent TrainState
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
        sac.target_network,
        sac.temperature,
        batch,
        gamma,
        is_discrete,
    )
    if update_target:
        new_target_critic = update_target_critic(new_critic, sac.target_network, tau)
    else:
        new_target_critic = sac.target_network

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
        jnp.ones((batch.actions.shape[0],)),
        {**critic_info, **actor_info, **alpha_info},
    )


def build_sample_action(actor_fn: Callable, is_discrete: bool, evaluate: bool):
    def sample_action(
        params: FrozenDict,
        observations: AgentObservation,
        key: PRNGKey,
    ) -> Tuple[PRNGKey, Array]:
        """sample agent action

        Args:
            params (FrozenDict): agent parameter
            observations (Array): agent observatoin
            key (PRNGKey): random key variable

        Returns:
            Tuple[PRNGKey, Array]: new key, sampled action
        """
        obs = observations.split_observation()
        if evaluate:
            if is_discrete:
                action_probs = actor_fn({"params": params}, obs)
                actions = jnp.argmax(action_probs, axis=-1)
            else:
                means, log_stds = actor_fn({"params": params}, obs)
                actions = means
        else:
            if is_discrete:
                action_probs = actor_fn({"params": params}, obs)
                action_dist = tfd.Categorical(probs=action_probs)
            else:
                means, log_stds = actor_fn({"params": params}, obs)
                action_dist = tfd.MultivariateNormalDiag(
                    loc=means, scale_diag=jnp.exp(log_stds)
                )

            subkey, key = jax.random.split(key)
            actions = action_dist.sample(seed=subkey)
        return key, actions

    return jax.jit(sample_action)


def restore_sac_actor(
    sac: SAC,
    is_discrete: bool,
    is_diff_drive: bool,
    restore_dir: str,
) -> SAC:
    """restore pretrained model

    Args:
        sac (SAC): Namedtuple of SAC agent composed by [Actor, Critic, TargetCritic, Temperature]
        is_discrete (bool): whether agent has discrete action space
        is_diff_drive (bool): whether agent has diff drive action space
        restore_dir (str): path to restore agent files in.

    Returns:
        SAC: restored sac
    """
    if not is_discrete:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=sac.actor,
            prefix="continuous_actor",
        ).params
    elif is_diff_drive:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=sac.actor,
            prefix="diff_drive_actor",
        ).params
    else:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=sac.actor,
            prefix="grid_actor",
        ).params
    actor = sac.actor.replace(params=actor_params)
    return sac._replace(actor=actor)
