"""utility for environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo

Credits: Code in this file is based on https://github.com/omron-sinicx/jaxmapp
"""

from functools import partial

import jax
import jax.numpy as jnp
from chex import Array


def xy_to_ij(pts, image_size: int) -> Array:
    return jnp.array(pts * image_size).astype(int)


@jax.jit
def _get_dist_to_edge(pts: Array, angle: Array, edge: Array) -> float:
    """
    get distance from a point to an edge.
    Main formulation is copied from https://github.com/danieldugas/pymap2d/blob/master/CMap2D.pyx

    Args:
        pts (Array): query point
        angle (Array): query angle
        edge (Array): edge

    Returns:
        float: _description_
    """
    a = jnp.cos(angle)
    b = -(edge[1, 1] - edge[0, 1])
    c = jnp.sin(angle)
    d = -(edge[1, 0] - edge[0, 0])
    e = edge[0, 1] - pts[1]
    f = edge[0, 0] - pts[0]
    det = a * d - b * c
    r_det = e * d - b * f
    t_det = -e * c + a * f
    r = r_det / (det + 1e-10)
    t = t_det / (det + 1e-10)
    cond = (det != 0) & (t > 0) & (t < 1) & (r >= 0)
    dist = jax.lax.cond(cond, lambda _: r, lambda _: jnp.inf, None)

    return dist


@jax.jit
def _get_dist_to_edges(pts: Array, angle: Array, edges: Array) -> Array:
    """
    get minimum distance from a point to a set of edges

    Args:
        pts (Array): query point
        angle (Array): query angle
        edges (Array): set of edges in the form of (n, 2, 2)

    Returns:
        Array: minimum distance to the set of edges
    """
    dist = jnp.min(
        jax.vmap(_get_dist_to_edge, in_axes=(None, None, 0))(pts, angle, edges)
    )
    return dist


@partial(jax.jit, static_argnames=("num_scans", "scan_range"))
def get_scans(
    pos: Array, rot: Array, edges: Array, num_scans: int, scan_range: float
) -> Array:
    """
    Simulate lidar scans

    Args:
        pos (Array): state position
        rot (Array): state rotation
        edges (Array): a set of edges extracted from occupancy map
        num_scans (int): number of scans
        scan_range (float): range of scans

    Returns:
        Array: scans where 0-th entry corresponds to the gaze direction
    """
    # angles = jax.vmap( lambda r: (jnp.linspace(0, 2 * jnp.pi, num_scans + 1)[:-1] + r) % (2 * jnp.pi))(rot)
    # scans = jax.vmap( jax.vmap(_get_dist_to_edges, in_axes=(None, 0, None)), in_axes=(0, 0, None),
    angles = (jnp.linspace(0, 2 * jnp.pi, num_scans + 1)[:-1] + rot) % (2 * jnp.pi)
    scans = jax.vmap(_get_dist_to_edges, in_axes=(None, 0, None))(pos, angles, edges)
    scans = jnp.minimum(scans, scan_range)
    return scans
