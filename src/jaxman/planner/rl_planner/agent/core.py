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
from .dqn.dqn import create_dqn_agent
from .sac.sac import SAC, _update_sac_jit
from .sac.sac import build_sample_action as build_sample_sac_action
from .sac.sac import create_sac_agent

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
    actor_fn: Callable, is_discrete: bool, evaluate: bool, model_name: str
):
    if model_name == "sac":
        return build_sample_sac_action(actor_fn, is_discrete, evaluate)
    else:
        return build_sample_dqn_action(actor_fn, evaluate)


@partial(
    jax.jit,
    static_argnames=(
        "is_discrete",
        "auto_temp_tuning",
        "update_target",
        "is_pal",
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
    is_pal: bool,
    alpha: bool,
    use_ddqn: bool,
    use_k_step_learning: bool,
    k: int,
    train_actor: bool,
    model_name: str,
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
        new_agent, priority, info = _update_dqn_jit(
            agent,
            batch,
            gamma,
            tau,
            is_pal,
            alpha,
            use_ddqn,
            use_k_step_learning,
            k,
        )
    return key, new_agent, priority, info
