""" vizualized discrete environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from __future__ import annotations

import io
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from chex import Array
from jaxman.env import AgentState, State, TrialInfo
from matplotlib import colormaps as cm

COLORS = cm.get_cmap("Set1").colors
GRAY = cm.get_cmap("Dark2").colors[-1]

### plot dot ###
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
    # render "x"
    image[x, y] = color
    image[x + 1, y + 1] = color
    image[x + 1, y - 1] = color
    image[x - 1, y + 1] = color
    image[x - 1, y - 1] = color
    return image


def plot_item(image, x, y, color, item_time=None):
    # render "+"
    if item_time is not None:
        color = np.array(color) * (
            1 - np.clip(np.array(item_time), a_min=0, a_max=100) / 200
        )
    image[x, y] = color
    image[x + 1, y] = color
    image[x - 1, y] = color
    image[x, y + 1] = color
    image[x, y - 1] = color
    return image


def plot_carried_item(image, x, y, color, item_time=None):
    if item_time is not None:
        color = np.array(color) * (
            1 - np.clip(np.array(item_time), a_min=0, a_max=100) / 200
        )
    image[x, y] = color
    return image


def render_agent_and_obstacles(
    state: AgentState,
    occupancy: Array,
    is_collided: Array = None,
    done: Array = None,
    is_diff_drive: bool = False,
    is_colorful: bool = False,
):
    """
    plot agent and obstacles

    Args:
        state (AgentState): agent current state
        occupancy (Array): Occunacy map
        is_collided (TrialInfo, optional): trial information
        done (Array, optional): done for each agent
        is_diff_drive (bool, optional): whether environment is diff drive env or not
        is_colorful (bool, optional): Are all agents the same color or are they colorful?
    """
    # resize
    occupancy = np.array(occupancy)
    occupancy = cv2.resize(
        occupancy, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
    )
    # goals = goals * 3 + 1
    pos = np.array(state.pos * 3 + 1, dtype=int)

    bg = np.ones((*occupancy.T.shape, 3), np.float32)
    bg[occupancy.T == 1.0, :] = [0.02, 0.05, 0.05]
    bg[occupancy.T == 0.0, :] = [0.98, 0.95, 0.95]

    # Agent
    for i in range(len(state.pos)):
        if is_colorful:
            color = COLORS[i % len(COLORS)]
        else:
            color = GRAY

        # bg = plot_goal(bg, goals[i, 1], goals[i, 0], color)
        if is_collided is not None:
            if done is not None:
                if done[i] and is_collided[i]:
                    bg = plot_collide_robot(bg, pos[i, 1], pos[i, 0], color)
                else:
                    if is_diff_drive:
                        bg = plot_diff_drive_robot(
                            bg, pos[i, 1], pos[i, 0], state.rot[i, 0], color
                        )
                    else:
                        bg = plot_grid_robot(bg, pos[i, 1], pos[i, 0], color)
            elif is_collided[i]:
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

    return bg


def render_navigation(image: np.ndarray, goals: Array) -> np.ndarray:
    """
    render navigation task
    this function is used with `render_agent_and_obstacles`

    Args:
        image (np.ndarray): agent and obstacle image
        goals (Array): agent goals

    Returns:
        np.ndarray: image of navigation task
    """
    # render agent goals
    goals = goals * 3 + 1
    for i in range(len(goals)):
        color = COLORS[i % len(COLORS)]
        image = plot_goal(image, goals[i, 1], goals[i, 0], color)

    return image


def render_pick_and_delivery(
    image: np.ndarray, state: State, goals: Array
) -> np.ndarray:
    """
    render pick and delivery task
    this function is used with `render_agent_and_obstacles`

    Args:
        image (np.ndarray): agent and obstacle image
        state (np.ndarray): agent and item state
        goals (Array): agent goals

    Returns:
        np.ndarray: image of navigation task
    """

    num_items = len(goals)
    agent_pos = state.agent_state.pos * 3 + 1
    item_pos = state.item_pos * 3 + 1
    item_goals = goals * 3 + 1

    for i in range(num_items):
        color = COLORS[i % len(COLORS)]

        is_carried = np.any(np.equal(i, state.load_item_id))
        if is_carried:
            # render carried items
            loaded_agent_id = np.where(np.equal(state.load_item_id, i))[0][0]
            image = plot_carried_item(
                image,
                int(agent_pos[loaded_agent_id, 1]),
                int(agent_pos[loaded_agent_id, 0]),
                color,
                state.item_time[i],
            )
        # if item is carried or collided, item_pos is set to INF
        elif item_pos[i][0] < image.shape[0]:
            # render not carried item
            image = plot_item(
                image, item_pos[i, 1], item_pos[i, 0], color, state.item_time[i]
            )

        # render item goal
        image = plot_goal(image, item_goals[i, 1], item_goals[i, 0], color)

    return image


def update_image_to_hihg_resolution(img: np.ndarray, occupancy: Array) -> np.ndarray:
    """update low-resolution images to high-resolution

    Args:
        img (np.ndarray): original low-resolution image
        occupancy (Array): occupancy map

    Returns:
        np.ndarray: updated high-resolution image
    """
    fig, axes = plt.subplots(1, 1, figsize=[10, 10])
    axes.imshow(img)
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


def render_env(
    state: Union[AgentState, State],
    goals: Array,
    occupancy: Array,
    trial_info: TrialInfo = None,
    done: Array = None,
    is_diff_drive: bool = False,
    is_high_resolution: bool = False,
    task_type: str = "navigation",
) -> np.ndarray:
    """render environment

    Args:
        state (Union[AgentState, State]): environment state
        goals (Array): goals
        occupancy (Array): obstacle occupancy map
        trial_info (TrialInfo, optional): trial information. Defaults to None.
        done (Array, optional): episode done for each agents. Defaults to None.
        is_diff_drive (bool, optional): whether agent action space is diff-drive or not. Defaults to False.
        is_high_resolution (bool, optional): whether update image to high-resolution. Defaults to False.
        task_type (str, optional): task type (navigation or pick and delivery). Defaults to "navigation".

    Returns:
        np.ndarray: environment image
    """
    is_navigation = task_type == "navigation"
    if is_navigation:
        agent_state = state
        is_collided = trial_info.collided
    else:
        agent_state = state.agent_state
        is_collided = trial_info.agent_collided

    # render agent and obstacle position
    img = render_agent_and_obstacles(
        agent_state,
        # goals,
        occupancy,
        is_collided,
        done,
        is_diff_drive,
        is_colorful=is_navigation,
    )
    if is_navigation:
        img = render_navigation(img, goals)
    else:
        img = render_pick_and_delivery(img, state, goals)

    if is_high_resolution:
        img = update_image_to_hihg_resolution(img, occupancy)

    return img
