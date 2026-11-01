import numpy as np

import chex
from dataclasses import field

import jax.numpy as jnp
import jax
from jax.typing import ArrayLike

from .polygon import Polygon, polygon_out_of_bounds, get_polygon_edges, convex_polygon_intersect, \
    fill_convex_polygon, get_polygon_bbox_len

# UNITS: shapes assemble into square of length 4

# coordinates are (x, y); +y is UP
# origin (0, 0) is the center

@chex.dataclass(frozen=True)
class State:
    pos: jax.Array # of center of mass, shape (n_shapes=7, n_dims=2)
    vel: jax.Array # of center of mass, shape (n_shapes=7, n_dims=2)
    ang: jax.Array # about center of mass, shape (n_shapes=7)
    ang_vel: jax.Array # about center of mass, shape (n_shapes=7)

@chex.dataclass(frozen=True)
class Action:
    target_vel: jax.Array # shape (n_shapes=7, n_dims=2)
    target_ang_vel: jax.Array #  shape (n_shapes=7)

@chex.dataclass(frozen=True)
class Settings:
    sim_dt: float = 0.025 # time (s) between physics steps
    ctrl_dt: float = 0.1 # time (s) between new actions; should be a multiple of sim_dt

    bounds: np.ndarray = field(default_factory=lambda: np.array(((-6, -6), (6, 6))))

    # max norm in any direction, same for each shape; be convervative!
    max_vel = 1
    max_accel = 1
    max_ang_vel = 1
    max_ang_accel = 1

@chex.dataclass(frozen=True)
class RenderSettings:
    scale: float = 32 # scale of 1 -> shapes assemble into square of length 4 

    background_color: np.ndarray = field(default_factory=lambda: np.array((255, 255, 255), dtype=np.uint8))

    shape_colors: np.ndarray = field(default_factory=lambda: np.array(( 
        # https://sashamaps.net/resources/20-colors/
        (230, 25, 75), # red
        (245, 130, 48), # orange
        (255, 225, 25), # yellow
        # (210, 245, 60), # lime
        (60, 180, 75), # green
        # (70, 240, 240), # cyan
        (0, 130, 200), # blue
        (145, 30, 180), # purple
        (240, 50, 230) # magenta
    ), dtype=np.uint8))

def _right_triangle_vertices(leg_len):
    v = np.array(((0, 0), (leg_len, 0), (0, leg_len))) # right triangle w/ legs along +x and +y
    v = v - v.mean(axis=0) # shift to make relative to center of mass
    return np.vstack((v, v[-1]))  # duplicate last vertex to pad 

class Sim:
    SHAPES: Polygon = Polygon( # NOTE: must be CONVEX polygons
        vertices = np.array(( # assembles into a square of length 4 (area 16)
            _right_triangle_vertices(np.sqrt(2)), # small triangle 1 (area 1)
            _right_triangle_vertices(np.sqrt(2)), # small triangle 2 (area 1)
            _right_triangle_vertices(2), # medium triangle (area 2)
            _right_triangle_vertices(2 * np.sqrt(2)), # large triangle 1 (area 4)
            _right_triangle_vertices(2 * np.sqrt(2)), # large triangle 2 (area 4)

            # square, side sqrt(2) (area 2)
            np.sqrt(2)/2 * np.array(((-1, -1), (1, -1), (1, 1), (-1, 1))),

            # parallelogram, sides 2 and sqrt(2) (area 2), center of mass centered
            np.array(((-1.5, 0.5), (0.5, 0.5), (1.5, -0.5), (-0.5, -0.5))),
        )),

        n_vertices = np.array((3, 3, 3, 3, 3, 4, 4))
    )

    def __init__(self, settings=Settings(), render_settings=RenderSettings()):
        self.settings = settings
        self.render_settings = render_settings

        sim_steps_per_ctrl = self.settings.ctrl_dt / self.settings.sim_dt
        assert np.isclose(sim_steps_per_ctrl, np.round(sim_steps_per_ctrl)), (
            "`settings.ctrl_dt` must be a multiple of `settings.sim_dt`")

    def reset(self) -> State:
        """Default starting position: assembled as a square, centered around (0, 0); no vel or ang_vel"""

        return State(
            pos = jnp.array(( 
                (5/3 + 0.1, -1), # small triangle 1 (bottom-right corner)
                (0, 2/3), # small triangle 2 (above center)
                (4/3 + 0.1, 4/3 + 0.1), # medium triangle (top-right corner)
                (-4/3 - 0.1, 0), # large triangle 1 (left)
                (0, -4/3 - 0.1), # large triangle 2 (bottom)
                (1 + 0.05, 0), # square (right of center)
                (-0.5, 1.5 + 0.05), # parallelogram (top)
            )),
            vel = jnp.zeros((7, 2)),
            ang = jnp.array((
                -1/8 * 2*jnp.pi, # small triangle 1
                1/8 * 2*jnp.pi, # small triangle 2
                4/8 * 2*jnp.pi, # medium triangle
                3/8 * 2*jnp.pi, # large triangle 1
                5/8 * 2*jnp.pi, # large triangle 2
                1/8 * 2*jnp.pi, # square
                0, # parallelogram
            )),
            ang_vel = jnp.zeros(7)
        )

    def step(self, state: State, action: Action) -> tuple[State, jax.Array]:
        # check for collisions and out of bounds
        failed = self.check_failed(state)
        
        # only update state if not failed
        return jax.lax.cond(failed, 
            lambda: state, 
            lambda: self.update_state(state, action)
        ), failed

    def check_failed(self, state: State) -> jax.Array:
        return self.check_out_of_bounds(state) | self.check_collided(state)

    def check_out_of_bounds(self, state: State) -> jax.Array:
        vmapped_out_of_bounds = jax.vmap(polygon_out_of_bounds, in_axes=(0, 0, 0, None))
        return jnp.any(vmapped_out_of_bounds(state.pos, state.ang, self.SHAPES, self.settings.bounds))

    def check_collided(self, state: State) -> jax.Array:
        shape_edges = jax.vmap(get_polygon_edges)(self.SHAPES, state.ang, state.pos)

        get_colls = jax.vmap(jax.vmap(convex_polygon_intersect, in_axes=(None, 0)), in_axes=(0, None))
        coll_matrix = get_colls(shape_edges, shape_edges)

        # mask out self-intersections
        coll_matrix *= 1 - np.eye(coll_matrix.shape[0])

        # coll_cols = jnp.argmax(coll_matrix, axis=1)
        # first_col_row = jnp.argmax(jnp.sum(coll_matrix, axis=1) > 0)
        # jax.debug.print("COLL {i}, {j}", i=first_col_row, j=coll_cols[first_col_row])

        return jnp.any(coll_matrix)

    def update_state(self, state: State, action: Action) -> State:
        sim_steps_per_ctrl = int(self.settings.ctrl_dt / self.settings.sim_dt)

        def scan_func(state, _):
            return self.sim_step(state, action), None
        state, _ = jax.lax.scan(scan_func, state, None, length=sim_steps_per_ctrl, 
            unroll=True) # use unrolling since sim steps are simple and low in number?

        return state     

    def sim_step(self, state: State, action: Action) -> State:
        ### update velocities first, then positions (semi-implicit Euler method) ###

        # update velocities based on actions
        max_accel = self.settings.max_accel * self.settings.sim_dt
        max_ang_accel = self.settings.max_ang_accel * self.settings.sim_dt

        dv = action.target_vel - state.vel
        dv_mag = jnp.linalg.norm(dv, axis=-1)
        accel_dir = dv / (dv_mag[..., None] + 1e-12)
        vel = state.vel + jnp.minimum(max_accel, dv_mag)[..., None] * accel_dir

        dw = action.target_ang_vel - state.ang_vel
        ang_vel = state.ang_vel + jnp.clip(dw, -max_ang_accel, max_ang_accel)

        # limit velocities
        vel_mag = jnp.linalg.norm(vel, axis=-1)
        vel_dir = vel / (vel_mag[..., None] + 1e-12)
        vel = jnp.minimum(self.settings.max_vel, vel_mag)[..., None] * vel_dir

        ang_vel = jnp.clip(ang_vel, -self.settings.max_ang_vel, self.settings.max_ang_vel)

        ### update state with new velocities ###
        state = state.replace(vel=vel, ang_vel=ang_vel)

        ### update positions ###
        state = state.replace(
            pos = state.pos + vel*self.settings.sim_dt, 
            ang = state.ang + ang_vel*self.settings.sim_dt
        )

        return state
    
    ### RENDERING ###
    @property
    def render_size(self):
        return (self.settings.bounds[1] - self.settings.bounds[0]) * self.render_settings.scale

    def render(self, state: State, action: Action | None = None) -> chex.Array:
        # need to add padding of maximum bbox len / 2 to ensure proper rendering
            # since the entire bbox must be within the image
        pad_amt = np.max(get_polygon_bbox_len(self.SHAPES, self.render_settings.scale)) // 2

        image_shape = (*(self.render_size + 2*pad_amt), 3)
        image = jnp.full(image_shape, self.render_settings.background_color, dtype=jnp.uint8)

        flattened_shapes, shapes_treedef = jax.tree.flatten(self.SHAPES)

        for pos, ang, color, *flattened_shape in zip(
                state.pos, state.ang, self.render_settings.shape_colors, *flattened_shapes):
            shape = jax.tree.unflatten(shapes_treedef, flattened_shape)

            draw_pos = pos - self.settings.bounds[0]
            draw_pos *= self.render_settings.scale
            draw_pos += pad_amt # adjust for padding

            image = fill_convex_polygon(image, color, draw_pos, ang, shape, self.render_settings.scale)

        return image[pad_amt:-pad_amt, pad_amt:-pad_amt]