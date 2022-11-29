"""vizualize environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from __future__ import annotations

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from chex import Array
from jaxman.env.core import AgentState, TrialInfo
from matplotlib import colormaps as cm

from .utils import xy_to_ij

COLORS = cm.get_cmap("Set1").colors


def plot_goal(image, x, y, color):
    image[x - 1, y - 1 : y + 2] = color
    image[x + 1, y - 1 : y + 2] = color
    image[x, y - 1] = color
    image[x, y + 1] = color
    return image


def plot_grid_robot(image, x, y, color):
    image[x - 1 : x + 2, y - 1 : y + 2] = color
    return image


def plot_diff_drive_robot(image, x, y, rot, color):
    image[x - 1 : x + 2, y - 1 : y + 2] = color
    if rot == 0:
        image[x + 1, y - 1] = [0, 0, 0]
        image[x + 1, y + 1] = [0, 0, 0]
    if rot == 1:
        image[x - 1, y + 1] = [0, 0, 0]
        image[x + 1, y + 1] = [0, 0, 0]
    if rot == 2:
        image[x - 1, y - 1] = [0, 0, 0]
        image[x - 1, y + 1] = [0, 0, 0]
    if rot == 3:
        image[x - 1, y - 1] = [0, 0, 0]
        image[x + 1, y - 1] = [0, 0, 0]
    return image


def plot_collide_robot(image, x, y, color):
    image[x, y] = color
    image[x + 1, y + 1] = color
    image[x + 1, y - 1] = color
    image[x - 1, y + 1] = color
    image[x - 1, y - 1] = color
    return image


def render_grid_map(
    state: AgentState,
    goals: Array,
    occupancy: Array,
    trial_info: TrialInfo = None,
    done: Array = None,
    is_diff_drive: bool = False,
    high_quality: bool = False,
):
    """
    Plot the environment

    Args:
        state (Array): agent current state
        goals (Array): Goal locations of agents
        occupancy (Array): Occunacy map
        trial_info (TrialInfo, optional): trial information
        done (Array, optional): done for each agent
        is_diff_drive (bool, optional): whether environment is diff drive env or not
        high_quality (bool, optional): whether to output high resolution rendering map
    """
    # resize
    occupancy = np.array(occupancy)
    occupancy = cv2.resize(
        occupancy, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
    )
    goals = goals * 3 + 1
    pos = np.array(state.pos * 3 + 1, dtype=int)

    bg = np.ones((*occupancy.T.shape, 3), np.float32)
    bg[occupancy.T == 1.0, :] = [0.02, 0.05, 0.05]
    bg[occupancy.T == 0.0, :] = [0.98, 0.95, 0.95]

    # Goal
    for i in range(len(goals)):
        color = COLORS[i % len(COLORS)]

        bg = plot_goal(bg, goals[i, 1], goals[i, 0], color)
        if trial_info is not None:
            if done is not None:
                if done[i] and trial_info.collided[i]:
                    bg = plot_collide_robot(bg, pos[i, 1], pos[i, 0], color)
                else:
                    if is_diff_drive:
                        bg = plot_diff_drive_robot(
                            bg, pos[i, 1], pos[i, 0], state.rot[i, 0], color
                        )
                    else:
                        bg = plot_grid_robot(bg, pos[i, 1], pos[i, 0], color)
            elif trial_info.collided[i]:
                bg = plot_collide_robot(bg, pos[i, 1], pos[i, 0], color)
            elif is_diff_drive:
                bg = plot_diff_drive_robot(
                    bg, pos[i, 1], pos[i, 0], state.rot[i, 0], color
                )
            else:
                bg = plot_grid_robot(bg, pos[i, 1], pos[i, 0], color)
        else:
            if is_diff_drive:
                bg = plot_diff_drive_robot(
                    bg, pos[i, 1], pos[i, 0], state.rot[i, 0], color
                )
            else:
                bg = plot_grid_robot(bg, pos[i, 1], pos[i, 0], color)
    if high_quality:
        fig, axes = plt.subplots(1, 1, figsize=[10, 10])
        axes.imshow(bg)
        map_size = occupancy.shape[0]
        axes.set_xlim([0, map_size])
        axes.set_ylim([map_size, 0])
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
    else:
        return bg


def render_simple_continuous_map(
    starts: Array,
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
):
    """
    render continous map with cv2 (low resolution)

    Args:
        starts (Array): Start locations of agents
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
        agent_pos = np.array(xy_to_ij(starts[i], image_size))
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


def render_map(
    state: AgentState,
    goals: Array,
    rads: Array,
    occupancy: Array,
    state_traj: Array = None,
    trial_info: TrialInfo = None,
    done: Array = None,
    is_discrete: bool = False,
    is_diff_drive: bool = False,
    high_quality: bool = True,
):
    if is_discrete:
        img = render_grid_map(
            state, goals, occupancy, trial_info, done, is_diff_drive, high_quality
        )
    else:
        starts = state.pos
        if high_quality:
            img = render_continuous_map(starts, goals, rads, occupancy, state_traj)
        else:
            img = render_simple_continuous_map(
                starts, goals, rads, occupancy, state_traj
            )
    return img


def render_gif(
    state_traj: Array,
    goals: Array,
    rads: Array,
    occupancy: Array,
    trial_info: TrialInfo = None,
    dones: Array = None,
    is_discrete: bool = False,
    is_diff_drive: bool = False,
    high_quality: bool = False,
):
    episode_steps = len(state_traj)
    full_obs = []

    for t in range(episode_steps):
        state = AgentState.from_array(state_traj[t])
        if dones is not None:
            done = dones[t]
        else:
            done = None
        if is_discrete:
            bg = render_grid_map(
                state, goals, occupancy, trial_info, done, is_diff_drive, high_quality
            )
        else:
            starts = state.pos
            if high_quality:
                bg = render_continuous_map(
                    starts, goals, rads, occupancy, state_traj[t:]
                )
            else:
                bg = render_simple_continuous_map(
                    starts, goals, rads, occupancy, state_traj[t:]
                )
        full_obs.append(bg.copy())

    return full_obs
