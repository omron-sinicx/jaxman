from typing import Union

from jaxman.env.pick_and_delivery.core import State

from .core import AgentInfo, AgentState, EnvInfo
from .navigation.core import AgentObservation as NaviObs
from .navigation.core import TaskInfo as NaviTaskInfo
from .navigation.core import TrialInfo as NaviTrialInfo
from .navigation.instance import Instance as NaviInstance
from .pick_and_delivery.core import AgentObservation as PandDObs
from .pick_and_delivery.core import TaskInfo as PandDTaskInfo
from .pick_and_delivery.core import TrialInfo as PandDTrialInfo
from .pick_and_delivery.instance import Instance as PandDInstance

AgentObservation = Union[NaviObs, PandDObs]
TaskInfo = Union[NaviTaskInfo, PandDTaskInfo]
TrialInfo = Union[NaviTrialInfo, PandDTrialInfo]
Instance = Union[NaviInstance, PandDInstance]
