"""AgentObservation for training agent network

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import NamedTuple

from chex import Array


class AgentObservation(NamedTuple):
    base_observation: Array
    communication: Array
    agent_mask: Array
    item_positions: Array = None
    item_mask: Array = None
