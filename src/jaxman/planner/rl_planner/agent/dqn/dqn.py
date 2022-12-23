"""Definition of create and update dqn agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Dict, Tuple, Union, NamedTuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax.training import checkpoints
from flax.training.train_state import TrainState
from gym.spaces import Box, Dict, Discrete
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig
from ...core import AgentObservation
from ..model.discrete_model import Critic
import optax
from .update import update
from ..sac.critic import update_target_critic

class DQN(NamedTuple):
    actor: TrainState
    target_actor: TrainState


def create_dqn_agent(
    observation_space: Dict,
    action_space: Union[Box, Discrete],
    config: DictConfig,
    key: PRNGKey,
) -> Tuple[TrainState, PRNGKey]:
    """create sac agent

    Args:
        observation_space (Dict): agent observation space
        action_space (Union[Box, Discrete]): agent action space
        model_config (DictConfig): model configurations
        key (PRNGKey): random variable key

    Returns:
        Tuple[TrainState,TrainState,TrainState,TrainState,PRNGKey]: SAC agent and key
    """
    actor_key, target_actor_key, key = jax.random.split(key,3)
    obs = jnp.ones([1, *observation_space["obs"].shape])
    comm = jnp.ones([1, *observation_space["comm"].shape])
    mask = jnp.ones([1, *observation_space["mask"].shape])
    if observation_space["item_pos"].shape[0] > 0:
        item_pos = jnp.ones([1, *observation_space["item_pos"].shape])
        item_mask = jnp.ones([1, *observation_space["item_mask"].shape])
    else:
        item_pos = item_mask = None
    dummy_observations = AgentObservation(obs, comm, mask, item_pos, item_mask)

    action_dim = action_space.n
    actor_fn = Critic(config.hidden_dim, config.msg_dim, action_dim)
    target_actor_fn = Critic(config.hidden_dim, config.msg_dim, action_dim)
    params = actor_fn.init(
            actor_key,
            dummy_observations,
        )["params"]
    target_params = target_actor_fn.init(target_actor_key, dummy_observations)["params"]
    lr_rate_schedule = optax.cosine_decay_schedule(
        config.actor_lr, config.decay_steps, config.decay_alpha
    )
    tx = optax.adam(learning_rate=lr_rate_schedule)
    target_tx = optax.adam(learning_rate=lr_rate_schedule)
    actor = TrainState.create(apply_fn=actor_fn.apply, params=params, tx=tx)
    target_actor = TrainState.create(apply_fn=target_actor_fn.apply, params=target_params, tx=target_tx)
    dqn = DQN(actor, target_actor)
    return dqn, key


# def restore_sac_actor(
#     actor: TrainState,
#     is_discrete: bool,
#     is_diff_drive: bool,
#     restore_dir: str,
# ) -> Tuple[TrainState, TrainState, TrainState, TrainState]:
#     """restore pretrained model

#     Args:
#         actor (TrainState): TrainState of actor
#         is_discrete (bool): whether agent has discrete action space
#         is_diff_drive (bool): whether agent has diff drive action space
#         restore_dir (str): path to restore agent files in.

#     Returns:
#         Tuple[TrainState]: restored actor
#     """
#     if not is_discrete:
#         actor_params = checkpoints.restore_checkpoint(
#             ckpt_dir=restore_dir,
#             target=actor,
#             prefix="continuous_actor",
#         ).params
#     elif is_diff_drive:
#         actor_params = checkpoints.restore_checkpoint(
#             ckpt_dir=restore_dir,
#             target=actor,
#             prefix="diff_drive_actor",
#         ).params
#     else:
#         actor_params = checkpoints.restore_checkpoint(
#             ckpt_dir=restore_dir,
#             target=actor,
#             prefix="grid_actor",
#         ).params
#     return actor.replace(params=actor_params)


@partial(
    jax.jit,
    static_argnames=(
        "is_discrete",
        "auto_temp_tuning",
        "update_target",
        "train_actor",
    ),
)
def _update_dqn_jit(
    key: PRNGKey,
    dqn: DQN,
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
    new_actor, actor_info = update(
        dqn.actor,
        dqn.target_actor,
        batch,
        gamma,
    )
    new_target_actor = update_target_critic(new_actor, dqn.target_actor, tau)
    new_dqn = DQN(new_actor, new_target_actor)
    return (
        key,
        new_dqn,
        actor_info,
    )
