import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxman.env.pick_and_delivery.core import AgentState, State, TaskInfo, TrialInfo
from jaxman.env.pick_and_delivery.env import JaxPandDEnv
from jaxman.utils import compute_agent_action
from omegaconf import OmegaConf

config = hydra.utils.instantiate(
    OmegaConf.load("scripts/config/env/pick_and_delivery/continuous.yaml")
)
config.level = 1
config.map_size = 128
config.num_agents = 3
config.num_items = 4
config.dist_reward = 0.05
config.is_discrete = False
env = JaxPandDEnv(config)
# plt.imshow(env.render())
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
obs = env.reset(subkey)

is_item_loaded = jnp.expand_dims(jnp.arange(env.num_items) < env.num_agents, -1)
item_starts = (env.task_info.item_starts + is_item_loaded * 10000).astype(int)
env.state = env.state._replace(
    load_item_id=jnp.arange(env.num_agents), item_pos=item_starts
)
fig, axes = plt.subplots(1, 10, figsize=(21, 3))
for i in range(10):
    key, subkey = jax.random.split(key)
    actions = jnp.array([[0.5, 0.2, 0], [0.1, -0.1, 0], [1.0, 1.0, 1]])
    # actions = compute_agent_action(["RIGHT","LEFT"])
    obs, rew, done, trial_info = env.step(actions)
    print(f"rew:{rew}, act:{actions}, done:{done}")
    axes[i].imshow(env.render())

# _observe(state, task_info, trial_info)
