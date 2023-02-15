""" initializer of rollout functio

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable

from jaxman.env import Instance

from .navigation.rollout import build_rollout_episode as build_navi_rollout
from .pick_and_delivery.rollout import build_rollout_episode as build_p_and_d_rollout


def _build_rollout_episode(
    instance: Instance,
    actor_fn: Callable,
    evaluate: bool,
    model_name: str,
) -> Callable:
    """build rollout episode function

    Args:
        instance (Instance): problem instance
        actor_fn (Callable): actor function
        evaluate (bool): whether agent explorate or evaluate

    Returns:
        Callable: jit-compiled rollout episode function
    """
    if instance.env_name == "navigation":
        return build_navi_rollout(instance, actor_fn, evaluate, model_name)
    else:
        return build_p_and_d_rollout(instance, actor_fn, evaluate, model_name)
