"""Definition of create and update DQN agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from gym.spaces import Dict, Discrete
from jaxman.env import AgentObservation
from jaxman.planner.rl_planner.memory.dataset import TrainBatch
from omegaconf import DictConfig

from ...core import AgentObservation
from ..model.discrete_model import Critic, MaxMinCritic, MultiHeadCritic
from ..sac.critic import update_target_critic
from .update import update


class DQN(NamedTuple):
    actor: TrainState
    target_network: TrainState


def create_dqn_agent(
    observation_space: Dict,
    action_space: Discrete,
    config: DictConfig,
    key: PRNGKey,
) -> Tuple[DQN, PRNGKey]:
    """create DQN agent

    Args:
        observation_space (Dict): agent observation space
        action_space (Discrete): agent action space
        model_config (DictConfig): model configurations
        key (PRNGKey): random variable key

    Returns:
        Tuple[DQN,PRNGKey]: DQN agent agent and key
    """
    actor_key, target_actor_key, key = jax.random.split(key, 3)
    obs = jnp.ones([1, *observation_space["obs"].shape])
    comm = jnp.ones([1, *observation_space["comm"].shape])
    mask = jnp.ones([1, *observation_space["mask"].shape])
    if observation_space["item_pos"].shape[0] > 0:
        item_pos = jnp.ones([1, *observation_space["item_pos"].shape])
        item_mask = jnp.ones([1, *observation_space["item_mask"].shape])
    else:
        item_pos = item_mask = None
    is_hold_item = jnp.ones([1])
    dummy_observations = AgentObservation(
        obs, comm, mask, item_pos, item_mask, is_hold_item
    )

    action_dim = action_space.n

    if config.use_multi_head:
        actor_fn = MultiHeadCritic(
            config.hidden_dim, config.msg_dim, action_dim, config.use_dueling_net
        )
        target_actor_fn = MultiHeadCritic(
            config.hidden_dim, config.msg_dim, action_dim, config.use_dueling_net
        )
    elif config.use_maxmin_dqn:
        actor_fn = MaxMinCritic(
            config.hidden_dim,
            config.msg_dim,
            action_dim,
            config.use_dueling_net,
            config.N,
        )
        target_actor_fn = MaxMinCritic(
            config.hidden_dim,
            config.msg_dim,
            action_dim,
            config.use_dueling_net,
            config.N,
        )

    else:
        actor_fn = Critic(
            config.hidden_dim, config.msg_dim, action_dim, config.use_dueling_net
        )
        target_actor_fn = Critic(
            config.hidden_dim, config.msg_dim, action_dim, config.use_dueling_net
        )
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
    target_actor = TrainState.create(
        apply_fn=target_actor_fn.apply, params=target_params, tx=target_tx
    )
    dqn = DQN(actor, target_actor)
    return dqn, key


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
def _update_dqn_jit(
    key: PRNGKey,
    dqn: DQN,
    batch: TrainBatch,
    gamma: float,
    tau: float,
    is_pal: bool,
    alpha: float,
    use_maxmin_dqn: bool,
    N: int,
    use_ddqn: bool,
    use_k_step_learning: bool,
    k: int,
) -> Tuple[PRNGKey, DQN, Array, Dict]:
    """
    update SAC agent network

    Args:
        key (PRNGKey): random variable key
        dqn (DQN): namedtuple of DQN agent
        batch (TrainBatch): Train Batch
        gamma (float): gamma. decay rate
        tau (float): tau. target critic update rate
        is_pal (bool): whether to use persistent advantage laerning or not
        alpha (float): weight of action gap used for persistent advantage learning
        use_maxmin_dqn (bool): whether to use Maxmin Q-Learning
        N (int): number of networks for Maxmin Q-Learning
        use_ddqn (bool): whether to use double dqn
        use_k_step_learning (bool): whether to use k step learning
        k (int): k for multi step learning

    Returns:
        Tuple[PRNGKey, DQN, Array, Dict]: random variable key, updated DQN agent, priority (td-error),loss informations
    """

    new_actor, priority, actor_info = update(
        key,
        dqn.actor,
        dqn.target_network,
        batch,
        gamma,
        is_pal,
        alpha,
        use_maxmin_dqn,
        N,
        use_ddqn,
        use_k_step_learning,
        k,
    )
    new_target_actor = update_target_critic(new_actor, dqn.target_network, tau)
    new_dqn = DQN(new_actor, new_target_actor)
    return (
        new_dqn,
        priority,
        actor_info,
    )


def build_sample_action(actor_fn: Callable, evaluate: bool, use_maxmin_dnq: bool):
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

        q_values = actor_fn({"params": params}, obs)
        if use_maxmin_dnq:
            q_values = jnp.min(q_values, axis=1)
        actions = jnp.argmax(q_values, axis=-1)
        if evaluate:
            pass
        else:
            key, key1, key2 = jax.random.split(key, 3)
            rand_action = jax.random.choice(key1, 6, shape=(1,))
            eps = jax.random.uniform(key2, shape=(1,))
            is_greedy = eps > 0.05
            actions = actions * is_greedy + rand_action * jnp.logical_not(is_greedy)
        return key, actions

    return jax.jit(sample_action)


def restore_dqn_actor(
    dqn: DQN,
    is_diff_drive: bool,
    model_config: DictConfig,
    restore_dir: str,
) -> DQN:
    """restore pretrained model

    Args:
        dqn (DQN): DQN agent
        is_diff_drive (bool): whether agent has diff drive action space
        model_config (DictConfig): model configuration
        restore_dir (str): path to restore agent files in.

    Returns:
        DQN: restored dqn agent
    """
    if is_diff_drive:
        actor_params = checkpoints.restore_checkpoint(
            ckpt_dir=restore_dir,
            target=dqn.actor,
            prefix="diff_drive_actor",
        ).params
    else:
        if model_config.use_multi_head:
            actor_params = checkpoints.restore_checkpoint(
                ckpt_dir=restore_dir,
                target=dqn.actor,
                prefix="grid_actor_multi_head",
            ).params
        elif model_config.use_maxmin_dqn:
            actor_params = checkpoints.restore_checkpoint(
                ckpt_dir=restore_dir,
                target=dqn.actor,
                prefix="grid_actor_maxmin",
            ).params
        else:
            actor_params = checkpoints.restore_checkpoint(
                ckpt_dir=restore_dir,
                target=dqn.actor,
                prefix="grid_actor_single",
            ).params
    actor = dqn.actor.replace(params=actor_params)
    target_network = dqn.target_network.replace(params=actor_params)

    return dqn._replace(actor=actor, target_network=target_network)
