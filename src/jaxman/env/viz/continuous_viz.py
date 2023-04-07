"""visualize continous environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""


import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from chex import Array
from jaxman.env.core import AgentState
from jaxman.env.pick_and_delivery.core import State
from matplotlib import colormaps as cm

from ..utils import xy_to_ij

COLORS = cm.get_cmap("Set1").colors
GRAY = cm.get_cmap("Dark2").colors[-1]


def render_navi_simple_continuous_map(
    state: AgentState,
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
):
    """
    render continous map with cv2 (low resolution)

    Args:
        state (Array): Start locations of agents
        goals (Array): Goal locations of agents
        rads (Array): radius of agents
        occupancy (Array): Occunacy map
        state_traj (Array, optional): state trajectories as solution for the problem. Defaults to None
    """

    num_agents = len(goals)
    image_size = occupancy.T.shape[0]
    bg = np.ones((*occupancy.T.shape, 3), np.float32)
    bg[occupancy.T == 1.0, :] = [0.02, 0.05, 0.05]
    bg[occupancy.T == 0.0, :] = [0.98, 0.95, 0.95]

    for i in range(num_agents):
        # Robot
        color = COLORS[i % len(COLORS)]
        agent_pos = np.array(xy_to_ij(state.pos[i], image_size))
        cv2.circle(
            bg,
            agent_pos,
            int(rads[i, 0] * image_size),
            color,
            -1,
        )

        # Goal
        color = COLORS[i % len(COLORS)]
        goal_pos = np.array(xy_to_ij(goals[i], image_size))
        cv2.circle(
            bg,
            np.array(goal_pos),
            int(rads[i, 0] * image_size),
            color,
            1,
        )

        # trajectory
        if state_traj is not None:
            state_traj = np.array(state_traj)
            for i in range(len(goals)):
                color = np.array(COLORS[i % len(COLORS)])
                pos_traj_ij = np.array(xy_to_ij(state_traj[:, i, :2], image_size))
                cv2.polylines(bg, [pos_traj_ij], False, color)
                agent_pos = np.array(xy_to_ij(state_traj[0, i, :2], image_size))

    return bg


def render_p_and_d_simple_continuous_map(
    state: State,
    goals: Array,
    rads: Array,
    occupancy: Array,
):
    """
    render continous map with cv2 (low resolution)

    Args:
        state (Array): Start locations of agents
        goals (Array): Goal locations of agents
        rads (Array): radius of agents
        occupancy (Array): Occunacy map
    """

    num_agents = len(state.agent_state.pos)
    num_items = len(goals)
    image_size = occupancy.T.shape[0]
    bg = np.ones((*occupancy.T.shape, 3), np.float32)
    bg[occupancy.T == 1.0, :] = [0.02, 0.05, 0.05]
    bg[occupancy.T == 0.0, :] = [0.98, 0.95, 0.95]
    for i in range(num_agents):
        # Robot
        agent_pos = np.array(xy_to_ij(state.agent_state.pos[i], image_size))
        ## position
        cv2.circle(
            bg,
            agent_pos,
            int(rads[i, 0] * image_size),
            GRAY,
            -1,
        )
        ## rotation
        x = state.agent_state.pos[i, 0] + 1.75 * rads[i, 0] * np.sin(
            state.agent_state.rot[i, 0]
        )
        y = state.agent_state.pos[i, 1] + 1.75 * rads[i, 0] * np.cos(
            state.agent_state.rot[i, 0]
        )
        agent_dir_pos = np.array(xy_to_ij(np.array([x, y]), image_size))
        cv2.line(bg, agent_pos, agent_dir_pos, GRAY, 1)

    for i in range(num_items):
        # Item
        color = COLORS[i % len(COLORS)]
        is_carried = np.any(np.equal(i, state.load_item_id))
        if is_carried:
            loaded_agent_id = np.where(np.equal(state.load_item_id, i))[0][0]
            item_pos = np.array(
                xy_to_ij(state.agent_state.pos[loaded_agent_id], image_size)
            )
            cv2.circle(
                bg,
                item_pos,
                int(0.5 * rads[0, 0] * image_size),
                color,
                -1,
            )
        else:
            item_pos = np.array(xy_to_ij(state.item_pos[i], image_size))
            cv2.circle(
                bg,
                item_pos,
                int(rads[0, 0] * image_size),
                color,
                -1,
            )
        # Goal
        color = COLORS[i % len(COLORS)]
        goal_pos = np.array(xy_to_ij(goals[i], image_size))
        cv2.circle(
            bg,
            np.array(goal_pos),
            int(rads[0, 0] * image_size),
            color,
            1,
        )

    return bg


def render_simple_continuous_map(
    state: State,
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
    task_type: str = "navigation",
) -> np.ndarray:
    if task_type == "navigation":
        img = render_navi_simple_continuous_map(
            state, goals, rads, occupancy, state_traj
        )
    else:
        img = render_p_and_d_simple_continuous_map(state, goals, rads, occupancy)
    return img


def render_continuous_map(
    starts: Array,
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
):
    """render continous map with pyplot (high resolution)

    Args:
        starts (Array): Start locations of agents
        goals (Array): Goal locations of agents
        rads (Array): radius of agents
        occupancy (Array): Occunacy map
        state_traj (Array, optional): state trajectories as solution for the problem. Defaults to None
    """
    fig, axes = plt.subplots(1, 1, figsize=[10, 10])
    axes.imshow(1 - occupancy.T, vmin=0, vmax=1, cmap="gray")
    scale = occupancy.shape[0]
    starts = xy_to_ij(starts, scale)
    goals = xy_to_ij(goals, scale)
    rads = (np.array(rads) * scale).astype(int)
    traj_xy = (
        (np.array(state_traj[:, :, :2]) * scale).astype(int)
        if state_traj is not None
        else None
    )

    for i in range(len(starts)):
        color = COLORS[i % len(COLORS)]
        axes.add_patch(plt.Circle(starts[i], rads[i], color=color))
        axes.add_patch(plt.Circle(goals[i], rads[i], color=color, fill=False))

        if state_traj is not None:
            axes.plot(traj_xy[:, i, 0], traj_xy[:, i, 1], "-o", color=color)
            axes.add_patch(plt.Circle(traj_xy[0, i], rads[i], color=color, alpha=0.7))

    axes.set_xlim([0, scale])
    axes.set_ylim([scale, 0])
    fig.tight_layout()
    axes.axis("off")

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    plt.close(fig)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()

    return img_arr
