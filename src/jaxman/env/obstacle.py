"""Obstacle map

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from chex import Array
from skimage.draw import disk


def contours_to_edges(contours: list) -> jnp.array:
    edges = []
    for contour in contours:
        edge_from = contour
        edge_to = jnp.vstack((contour[1:], contour[0:1]))
        edges.append(
            jnp.stack([jnp.vstack((f, t)) for f, t in zip(edge_from, edge_to)])
        )
    edges = jnp.vstack(edges)

    return edges


class ObstacleMap(NamedTuple):
    """Obstacle map containing occupancy map, signed distance function, and their contour"""

    occupancy: Array
    sdf: Array
    edges: Array
    padding_occupancy: Array = None

    def get_size(self):
        return self.occupancy.shape[0]


class ObstacleSphere(NamedTuple):
    """Static sphere obstacle"""

    pos: np.ndarray  # center position
    rad: float  # radius

    def draw_2d(self, map_size: int) -> np.ndarray:
        """
        Draw 2d image for creating occupancy maps

        Args:
            map_size (int): map size

        Returns:
            np.ndarray: occupancy map for the given circle obstacle
        """

        shape = (map_size, map_size)
        img = np.zeros(shape, dtype=np.float32)
        X = int(map_size * self.pos[0])
        Y = int(map_size * self.pos[1])
        R = int(map_size * self.rad)
        rr, cc = disk((X, Y), R, shape=shape)
        img[rr, cc] = 1.0

        return img


def cross_road(map_size: int, num_road: int):
    delta = int(map_size / (num_road + 1))
    road_width = int(map_size / 25)

    image = jnp.ones((map_size, map_size))
    for i in range(num_road):
        pos = delta * (i + 1)
        image[pos - road_width : pos + road_width, :] = 0
        image[:, pos - road_width : pos + road_width] = 0

    # add frame
    image[:, 0] = 1
    image[:, -1] = 1
    image[0, :] = 1
    image[-1, :] = 1
    return image


def room(map_size: int, level: int) -> jnp.ndarray:
    image = jnp.zeros((map_size, map_size))
    delta = int(map_size / (level))
    door_width = int(map_size / 30)
    door_width2 = int(map_size / 25)
    if level == 1:
        door_width = int(map_size / 35)
        door_width2 = int(map_size / 25)
        center = int(map_size / 2)
        image[center - 1 : center + 1, :] = 1

        image[:, :door_width2] = 0
        image[:, center - door_width2 : center + door_width2] = 0
        image[:, -door_width2:] = 0
    else:
        for i in range(level - 1):
            pos = delta * (i + 1)
            image[pos - 1 : pos + 1, :] = 1
            image[:, pos - 1 : pos + 1] = 1
        for i in range(level + 1):
            pos0 = delta * (i + 1)
            door_pos = int(delta * (i + 0.5))
            image[door_pos - door_width : door_pos + door_width, :] = 0
            image[:, door_pos - door_width : door_pos + door_width] = 0
            for j in range(level - 1):
                if i < (level - 1):
                    pos1 = delta * (j + 1)
                    image[
                        pos0 - door_width2 : pos0 + door_width2,
                        pos1 - door_width2 : pos1 + door_width2,
                    ] = 0

    # add frame
    image[:, 0] = 1
    image[:, -1] = 1
    image[0, :] = 1
    image[-1, :] = 1
    return image
