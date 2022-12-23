import jax.numpy as jnp
from jaxman.env.core import AgentInfo, AgentState, EnvInfo, ObstacleMap
from jaxman.env.pick_and_delivery.core import State, TaskInfo, TrialInfo
from jaxman.env.pick_and_delivery.dynamics import _build_inner_step
from jaxman.env.viz.viz import render_env

obs = ObstacleMap(jnp.zeros((10, 10)), None, None, None)
env_info = EnvInfo(
    num_agents=2,
    occupancy_map=jnp.zeros((10, 10)),
    sdf_map=None,
    edges=None,
    num_items=2,
    fov_r=3,
    comm_r=3,
    num_scans=None,
    scan_range=None,
    use_intentions=True,
    timeout=100,
    goal_reward=+1,
    crash_penalty=-1,
    time_penalty=0,
    is_discrete=True,
    is_diff_drive=False,
)
zero = jnp.array([[0], [0]])
agent_info = AgentInfo(
    max_vels=zero,
    min_vels=zero,
    max_ang_vels=zero,
    min_ang_vels=zero,
    max_accs=zero,
    max_ang_accs=zero,
    rads=zero,
)
trial_info = TrialInfo.reset(env_info.num_agents, env_info.num_items)
task_info = TaskInfo(
    starts=jnp.array([[3, 3], [6, 6]]),
    start_rots=jnp.array([[0], [0]]),
    item_starts=jnp.array([[3, 6], [6, 3]]),
    item_goals=jnp.array([[7, 4], [4, 7]]),
    obs=obs,
)
agent_state = AgentState(
    pos=task_info.starts,
    rot=task_info.start_rots,
    vel=jnp.array([[0], [0]]),
    ang=jnp.array([[0], [0]]),
)
state = State(
    agent_state,
    load_item_id=jnp.ones((env_info.num_agents), dtype=int) * env_info.num_items,
    item_pos=task_info.item_starts,
)
done = jnp.array([False] * env_info.num_agents)

_step = _build_inner_step(env_info, agent_info)

render_env(
    state,
    task_info.item_goals,
    env_info.occupancy_map,
    trial_info,
    done,
    task_type="pick_and_delivery",
)

actions = jnp.array([2, 0])
state, rew, done, info = _step(state, actions, task_info, trial_info)
