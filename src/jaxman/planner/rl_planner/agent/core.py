"""common core architectures for DQN and SAC agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable, Dict, Tuple, Union

import jax
from chex import PRNGKey
from flax.training.train_state import TrainState
from gym.spaces import Box, Discrete
from omegaconf import DictConfig

from ..memory.dataset import TrainBatch
from .dqn.dqn import DQN, _update_dqn_jit
from .dqn.dqn import build_sample_action as build_sample_dqn_action
from .dqn.dqn import create_dqn_agent, restore_dqn_actor
from .sac.sac import SAC, _update_sac_jit
from .sac.sac import build_sample_action as build_sample_sac_action
from .sac.sac import create_sac_agent, restore_sac_actor

Agent = Union[SAC, DQN]


def create_agent(
    observation_space: Dict,
    action_space: Union[Box, Discrete],
    model_config: DictConfig,
    key: PRNGKey,
) -> Tuple[Agent, PRNGKey]:
    """create sac agent

    Args:
        observation_space (Dict): agent observation space
        action_space (Union[Box, Discrete]): agent action space
        model_config (DictConfig): model configurations
        key (PRNGKey): random variable key

    Returns:
        Tuple[TrainState,TrainState,TrainState,TrainState,PRNGKey]: SAC agent and key
    """
    if model_config.name == "sac":
        agent, key = create_sac_agent(
            observation_space, action_space, model_config, key
        )
    else:
        agent, key = create_dqn_agent(
            observation_space, action_space, model_config, key
        )
    return agent, key


def build_sample_agent_action(
    actor_fn: Callable,
    is_discrete: bool,
    env_name: str,
    evaluate: bool,
    model_config: DictConfig,
):
    if model_config.name == "sac":
        return build_sample_sac_action(actor_fn, is_discrete, env_name, evaluate)
    else:
        return build_sample_dqn_action(actor_fn, evaluate)


@partial(
    jax.jit,
    static_argnames=(
        "is_discrete",
        "auto_temp_tuning",
        "update_target",
        "use_ddqn",
        "use_k_step_learning",
        "train_actor",
        "model_name",
    ),
)
def _update_jit(
    key: PRNGKey,
    agent: Agent,
    batch: TrainBatch,
    gamma: float,
    tau: float,
    is_discrete: bool,
    target_entropy: float,
    auto_temp_tuning: bool,
    update_target: bool,
    use_ddqn: bool,
    use_k_step_learning: bool,
    k: int,
    train_actor: bool,
    model_name: str,
) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]:
    """
    update agent network

    Args:
        key (PRNGKey): random varDable key
        agent (Agent): Namedtuple of agent.
        batch (TrainBatch): Train Batch
        gamma (float): gamma. decay rate
        tau (float): tau. target critic update rate
        is_discrete (bool): whether agent action space is Discrete or not
        target_entropy (float): target entropy
        auto_temp_tuning (bool): whether to update temperature
        update_target (bool): whether to update target_critic network
        use_ddqn (bool): whether to use double dqn
        use_k_step_learning (bool): whether to use k step learning
        k (int): k for multi step learning
        train_actor (bool): whether to update actor
        model_name (str): model name, sac or dqn

    Returns:
        Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]: new key, updated SAC agent, loss informations
    """
    if model_name == "sac":
        key, new_agent, priority, info = _update_sac_jit(
            key,
            agent,
            batch,
            gamma,
            tau,
            is_discrete,
            target_entropy,
            auto_temp_tuning,
            update_target,
            train_actor,
        )
    else:
        key, subkey = jax.random.split(key)
        new_agent, priority, info = _update_dqn_jit(
            subkey,
            agent,
            batch,
            gamma,
            tau,
            use_ddqn,
            use_k_step_learning,
            k,
        )
    return key, new_agent, priority, info


def restore_agent(
    agent: Agent,
    is_discrete: bool,
    is_diff_drive: bool,
    model_config: DictConfig,
    restore_dir: str,
) -> Agent:
    if isinstance(agent, SAC):
        return restore_sac_actor(agent, is_discrete, is_diff_drive, restore_dir)
    else:
        return restore_dqn_actor(agent, is_diff_drive, model_config, restore_dir)
