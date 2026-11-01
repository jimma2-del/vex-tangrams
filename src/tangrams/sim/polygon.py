import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import chex

def rotation_matrix_2d(ang):
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    return jnp.array(((cos, -sin), (sin, cos)))

@chex.dataclass(frozen=True)
class Polygon:
    vertices: ArrayLike # shape (n_vertices, n_dims=2)
        # positions are offsets from the center of mass
        # we duplicate the last vertex to pad the array for shapes with fewer vertices
    
    n_vertices: np.ndarray

def transform_vertices(polygon: Polygon, ang: ArrayLike=0, pos: ArrayLike=jnp.zeros(2)) -> jax.Array:
    """Rotates the polygon vertices by `ang` about (0, 0) and translates by `pos`."""
    
    vertices = polygon.vertices

    # rotate vertices
    vertices = rotation_matrix_2d(ang) @ vertices.T
    vertices = vertices.T # transform back to shape (n_vertices, n_dims=2)

    # translate vertices
    vertices += pos

    return vertices

def polygon_out_of_bounds(pos: ArrayLike, ang: ArrayLike, polygon: Polygon, bounds: ArrayLike) -> jax.Array:
    vertices = transform_vertices(polygon, ang, pos) # (n_vertices, n_dims=2)
    return jnp.any((vertices < bounds[0]) | (vertices > bounds[1]), axis=(-2, -1))

def get_polygon_edges(polygon: Polygon, ang: ArrayLike=0, pos: ArrayLike=jnp.zeros(2)) -> jax.Array:
    """Creates a list of edge startpoints and a list of edge endpoints for the given vertices.
    Returns: start, end as a single JAX array of shape (2, n_vertices, n_dims=2)"""
    
    start_vertices = transform_vertices(polygon, ang, pos)

    # end vertex is the PREVIOUS vertex to allow extra padding edges to be easily masked out
    end_vertices = jnp.roll(start_vertices, 1, axis=0)

    return jnp.stack((start_vertices, end_vertices))

def convex_polygon_intersect(polygon_edges1: jax.Array, polygon_edges2: jax.Array) -> jax.Array:
    start1, end1 = polygon_edges1 # (N, n_dims=2)
    start2, end2 = polygon_edges2 # (M, n_dims=2)

    # get axes to test: the normal of every edge of both polygons
    edges = jnp.concatenate((end1 - start1, end2 - start2)) # (N+M, 2)
    axes = jnp.stack((-edges[:, 1], edges[:, 0]), axis=-1) # (N+M, 2)
        # perpendicular: -y, x

    # project each polygon's vertices onto every axis
    p1 = start1 @ axes.T # (N, N+M)
    p2 = start2 @ axes.T # (M, N+M)

    # an axis separates the polygons if their projected intervals don't overlap
    separated = (jnp.max(p1, axis=0) < jnp.min(p2, axis=0)) | (jnp.max(p2, axis=0) < jnp.min(p1, axis=0))

    # intersect only if no separating axis exists
    return ~separated.any()

def point_in_convex_polygon(point: ArrayLike, polygon_edges: jax.Array) -> jax.Array:
    start_vertices, end_vertices = polygon_edges

    start_to_end = end_vertices - start_vertices
    start_to_point = point - start_vertices

    cross_prod = start_to_end[:, 0]*start_to_point[:, 1] - start_to_end[:, 1]*start_to_point[:, 0] 
        # cross product gives signed area; sign indicates which side the point is on
    
    # if point is on the same side of every line, it is inside the convex polygon
    return jnp.all(cross_prod >= 0) | jnp.all(cross_prod <= 0) # all to the left or all to the right

def get_polygon_bbox_len(polygon: Polygon, scale: float = 1) -> int:
    """Find the minimum static rectangular bounding box size to fit any dynamic rotaton.
    
    As the polygon rotates, it sweeps out a circle around the center (0, 0) 
        with radius equal to the maximum distance from the center to any vertex.
    The rect bounding box for this circle is thus a square with side length 2 * this radius.
    """
    max_dist = np.max(np.linalg.norm(polygon.vertices, axis=-1), axis=-1)
    return 2 * (np.floor(max_dist * scale).astype(int) + 1) 
        # floor + 1 is always inclusive of both endpoints

def fill_convex_polygon(
    image: ArrayLike, 
    color: ArrayLike, 
    pos: ArrayLike, 
    ang: ArrayLike, 
    polygon: Polygon, 
    scale: ArrayLike = 1,
    bb_len: float | None = None
) -> jax.Array:
    """
    `bb_len` needs to be STATIC.
        If `bb_len` is None, `polygon` and `scale` must be STATIC to allowing computing a STATIC `bb_len`.
    """

    if bb_len is None:
        bb_len = get_polygon_bbox_len(polygon, scale)

    # create grid of points inside rect bounding box, centered around (0, 0)
    points = jnp.mgrid[:bb_len, :bb_len] - bb_len//2 # shape (n_dims, bb_len, bb_len)
    points /= scale # convert back to regular units for intersection detection

    # create mask of grid points that are inside the rotated polygon
    intersects = jax.vmap(jax.vmap(point_in_convex_polygon, in_axes=(1, None)), in_axes=(1, None))
    in_polygon = intersects(points, get_polygon_edges(polygon, ang)) # vmap over x and y

    #jax.debug.print("start={start} end={end}", start=points[:,0,0], end=points[:,-1,-1])
    #in_polygon = jnp.ones_like(in_polygon)

    # apply polygon onto rect section of image
    patch_start_is = pos - bb_len//2
    patch_start_is = jnp.round(patch_start_is).astype(jnp.int32)
    patch_start_is = jnp.append(patch_start_is, 0) # add starting index of color channel

    cur_patch = jax.lax.dynamic_slice(image, patch_start_is, (bb_len, bb_len, 3))
    new_patch = jnp.where(in_polygon[:, :, None], color[None, None, :], cur_patch)

    # apply rectanglar section with polygon back onto the original image
    return jax.lax.dynamic_update_slice(image, new_patch, patch_start_is)